"""
Generic rollup helpers for quantification backends.
"""

import logging as _logging
from collections.abc import (
    Mapping as _Mapping,
    Sequence as _Sequence,
)
from typing import Any as _Any

import numpy as _np
import pandas as _pd
from pyspark.sql import (
    Column as _Column,
    DataFrame as _DataFrame,
    functions as _fns,
)
from pyspark.sql.types import (
    DoubleType as _DoubleType,
    LongType as _LongType,
    StructField as _StructField,
    StructType as _StructType,
)

_logger = _logging.getLogger(__name__)

from .utils import (
    _ReductionLike,
    _aggregate_reduced_columns,
    _filter_rollup_dataset,
    _join_aggregates,
    _normalize_intensity_column_map,
    _normalize_rollup_axes,
)


def _build_directlfq_schema(
    dataset: _Any,
    *,
    final_group_key_columns: _Sequence[str],
    intensity_output_columns: _Sequence[str],
) -> _StructType:
    fields = []
    for column_name in final_group_key_columns:
        data_type = (
            _LongType()
            if column_name == "__entity_id"
            else dataset.data.schema[column_name].dataType
        )
        fields.append(
            _StructField(
                column_name,
                data_type,
                nullable=True,
            )
        )
    fields.extend(
        _StructField(
            output_column,
            _DoubleType(),
            nullable=True,
        )
        for output_column in intensity_output_columns
    )
    return _StructType(fields)


def _make_feature_id(
    pdf: _pd.DataFrame,
    feature_key_columns: _Sequence[str],
) -> _pd.Series:
    columns = list(feature_key_columns)
    if not columns:
        raise ValueError(
            "feature_key_columns must contain at least one column"
        )

    if len(columns) == 1:
        return pdf[columns[0]].astype(str)

    return (
        pdf[columns]
        .fillna("")
        .astype(str)
        .agg(lambda values: "\x1f".join(values), axis=1)
    )


def _estimate_directlfq_track(
    pdf: _pd.DataFrame,
    *,
    sample_column: str,
    intensity_column: str,
    intensity_output_column: str,
    directlfq_log_level: int,
    entity_label: dict[str, _Any],
) -> _pd.DataFrame:
    result_columns = ["__entity_id", sample_column, intensity_output_column]

    if pdf.empty:
        return _pd.DataFrame(columns=result_columns)

    wide = pdf.pivot_table(
        index=["__entity_id", "__feature_id"],
        columns=sample_column,
        values=intensity_column,
        aggfunc="first",
        fill_value=None,
    )

    if wide.empty:
        return _pd.DataFrame(columns=result_columns)

    wide.replace(0, _np.nan, inplace=True)

    wide_is_neg = wide < 0
    if (wide_is_neg).any().any():
        _logger.warning(
            "Negative intensity values found for %s",
            entity_label,
        )
        wide[wide_is_neg] = _np.nan

    if wide.notna().sum().sum() == 0:
        return _pd.DataFrame(columns=result_columns)

    from directlfq import config as _lfq_config

    _lfq_config.setup_logging = lambda *_, **__: ()
    _lfq_config.check_wether_to_copy_numpy_arrays_derived_from_pandas()

    if not _lfq_config.COPY_NUMPY_ARRAYS_DERIVED_FROM_PANDAS:
        if not wide.to_numpy(copy=False).flags.writeable:
            _lfq_config.COPY_NUMPY_ARRAYS_DERIVED_FROM_PANDAS = True

    from directlfq.protein_intensity_estimation import (
        estimate_protein_intensities,
    )

    _logging.getLogger("directlfq").setLevel(directlfq_log_level)

    wide = _np.log2(wide)
    wide.index.set_names(
        [_lfq_config.PROTEIN_ID, _lfq_config.QUANT_ID],
        inplace=True,
    )
    _lfq_config.set_compile_normalized_ion_table(False)

    protein_df, _ = estimate_protein_intensities(
        wide,
        min_nonan=1,
        num_samples_quadratic=10,
        num_cores=1,
    )

    if protein_df.empty:
        return _pd.DataFrame(columns=result_columns)

    protein_df.rename(
        columns={_lfq_config.PROTEIN_ID: "__entity_id"},
        inplace=True,
    )

    protein_long = (
        protein_df.set_index("__entity_id")
        .stack()
        .reset_index()
        .rename(columns={"level_1": sample_column, 0: intensity_output_column})
    )

    return protein_long[result_columns]


def _estimate_directlfq_partition(
    pdf: _pd.DataFrame,
    *,
    sample_column: str,
    feature_key_columns: _Sequence[str],
    intensity_column_map: _Sequence[tuple[str, str]],
    directlfq_log_level: int,
) -> _pd.DataFrame:
    output_columns = [
        output_column for _, output_column in intensity_column_map
    ]
    result_columns = ["__entity_id", sample_column, *output_columns]

    if pdf.empty:
        return _pd.DataFrame(columns=result_columns)

    entity_id = pdf["__entity_id"].iloc[0]
    prepared_pdf = pdf.assign(
        __feature_id=_make_feature_id(pdf, feature_key_columns),
    )

    track_series = []
    for source_column, output_column in intensity_column_map:
        track_result = _estimate_directlfq_track(
            prepared_pdf,
            sample_column=sample_column,
            intensity_column=source_column,
            intensity_output_column=output_column,
            directlfq_log_level=directlfq_log_level,
            entity_label={"__entity_id": entity_id},
        )
        if track_result.empty:
            continue
        track_series.append(
            track_result.set_index(["__entity_id", sample_column])[
                output_column
            ].rename(output_column)
        )

    if not track_series:
        return _pd.DataFrame(columns=result_columns)

    merged_tracks = _pd.concat(
        track_series, axis=1, join="outer"
    ).reset_index()
    for output_column in output_columns:
        if output_column not in merged_tracks.columns:
            merged_tracks[output_column] = _np.nan

    result = merged_tracks[result_columns].copy()

    # Normalize returned sample values to plain Python-object dtype before
    # Spark/Arrow serializes the pandas UDF result.
    result[sample_column] = result[sample_column].astype(object)

    return result


def roll_up_directlfq(
    dataset: _Any,
    *,
    entity_key_columns: _Sequence[str],
    sample_column: str,
    feature_key_columns: _Sequence[str] | None = None,
    intensity_columns: _Mapping[str, str] | _Sequence[str] | str | None = None,
    preserved_column_reductions: _Mapping[str, _ReductionLike] | None = None,
    qvalue_threshold: float | None = None,
    filter_column: str | _Column | None = None,
) -> _DataFrame:
    """
    Roll up one or more intensity tracks to a final output grain using the
    DirectLFQ estimator within each final entity.

    This helper returns one row per ``(entity_key_columns, sample_column)``
    combination after any requested filtering is applied. ``feature_key_columns``
    define the lower-level feature identity that should be treated as the same
    measurable feature across samples within each final entity. DirectLFQ uses
    those feature keys to build the per-entity wide matrix passed to the
    estimator, with ``sample_column`` defining the sample axis and
    ``feature_key_columns`` defining the row identity.

    To support multi-track rollups, ``intensity_columns`` may be either a
    single source column, a sequence of source columns, or a mapping from
    source column name to output column name. Each requested track is estimated
    independently for the same final output grain and emitted as a separate
    column in the returned :py:class:`pyspark.sql.DataFrame`.

    Parameters
    ----------
    dataset
        Input dataset containing the source intensities and all grouping
        columns. If ``qvalue_threshold`` is specified, this must be a
        :py:class:`ConfidenceDataset`.
    entity_key_columns
        Columns identifying the final rollup entity, excluding the sample axis.
        The returned frame will contain one row per
        ``(entity_key_columns, sample_column)`` pair.
    sample_column
        Column identifying sample membership in the input dataset and output
        frame. This column must not also appear in ``entity_key_columns`` or
        ``feature_key_columns``.
    feature_key_columns
        Columns identifying the same lower-level feature across samples within
        each final entity. These keys are combined into the DirectLFQ feature
        ID used internally to pivot each entity partition to wide form. This
        parameter is required for the DirectLFQ backend.
    intensity_columns
        Source intensity column or columns to estimate. When a mapping is
        provided, keys are source column names and values are output column
        names. If omitted, ``dataset.intensity_column`` is used and the source
        column name is preserved.
    preserved_column_reductions
        Optional mapping of non-key input columns to reduction names or
        callables. Each preserved column is reduced at the same
        ``(entity_key_columns, sample_column)`` grain and included in the
        returned frame.
    qvalue_threshold
        Optional confidence threshold applied before rollup. When provided,
        only rows with ``dataset.qvalues <= qvalue_threshold`` are retained.
    filter_column
        Optional additional Spark filter applied before rollup. May be either a
        column name or a Spark boolean expression.

    Returns
    -------
    DataFrame
        A Spark DataFrame containing one row per
        ``(entity_key_columns, sample_column)`` pair, with one estimated output
        column per requested intensity track and any requested preserved
        columns.
    """
    filtered = _filter_rollup_dataset(
        dataset,
        qvalue_threshold=qvalue_threshold,
        filter_column=filter_column,
    )
    entity_keys, feature_keys, output_group_keys = _normalize_rollup_axes(
        filtered,
        entity_key_columns=entity_key_columns,
        sample_column=sample_column,
        feature_key_columns=feature_key_columns,
        require_feature_keys=True,
    )
    intensity_column_map = _normalize_intensity_column_map(
        filtered,
        intensity_columns,
    )
    directlfq_log_level = _logging.getLogger("directlfq").getEffectiveLevel()

    schema = _build_directlfq_schema(
        filtered,
        final_group_key_columns=["__entity_id", sample_column],
        intensity_output_columns=[
            output_column for _, output_column in intensity_column_map
        ],
    )
    entity_map = (
        filtered.data.select(*entity_keys)
        .distinct()
        .withColumn("__entity_id", _fns.monotonically_increasing_id())
    )
    directlfq_input = filtered.data.join(
        entity_map, on=entity_keys, how="inner"
    ).select(
        "__entity_id",
        sample_column,
        *feature_keys,
        *[source_column for source_column, _ in intensity_column_map],
    )
    intensities = directlfq_input.groupBy("__entity_id").applyInPandas(
        lambda pdf: _estimate_directlfq_partition(
            pdf,
            sample_column=sample_column,
            feature_key_columns=feature_keys,
            intensity_column_map=intensity_column_map,
            directlfq_log_level=directlfq_log_level,
        ),
        schema,
    )
    intensities = intensities.join(
        entity_map, on="__entity_id", how="inner"
    ).select(
        *output_group_keys,
        *[output_column for _, output_column in intensity_column_map],
    )

    preserved = _aggregate_reduced_columns(
        filtered,
        group_key_columns=output_group_keys,
        column_reductions=preserved_column_reductions,
    )

    return _join_aggregates(
        intensities,
        preserved,
        join_columns=output_group_keys,
    )
