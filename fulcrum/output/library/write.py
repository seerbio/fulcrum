"""
`fulcrum.output.library.write` -- implements overall workflow module
"""

import logging as _logging
import re as _re
from typing import (
    cast as _cast,
    Any as _Any,
    Callable as _Callable,
    Dict as _Dict,
    Optional as _Optional,
    Union as _Union,
)

import csv as _csv
import fsspec as _fsspec
import pandas as _pd
from pyspark.sql import (
    Column as _Column,
    DataFrame as _DataFrame,
    functions as _fns,
)

from wheely.mammoth import (
    PsmDataset as _PsmDataset,
    ConfidenceDataset as _ConfidenceDataset,
)
from wheely.mammoth.spectra import (
    SpectraDataset as _SpectraDataset,
)
from wheely.mammoth.spectra.parsers.registry import (
    get_backend as _get_spectra_backend,
)
from wheely.mammoth.spectra.utils import (
    peaklist_to_pairs as _peaklist_to_pairs,
)

from fulcrum.output.util import filter_psms

_logger = _logging.getLogger(__name__)


#: Monoisotopic residue masses (Da), used to compute theoretical b/y fragment ions for annotation.
_AA_RESIDUE_MASS = {
    "G": 57.02146,
    "A": 71.03711,
    "S": 87.03203,
    "P": 97.05276,
    "V": 99.06841,
    "T": 101.04768,
    "C": 103.00919,
    "L": 113.08406,
    "I": 113.08406,
    "N": 114.04293,
    "D": 115.02694,
    "Q": 128.05858,
    "K": 128.09496,
    "E": 129.04259,
    "M": 131.04049,
    "H": 137.05891,
    "F": 147.06841,
    "R": 156.10111,
    "Y": 163.06333,
    "W": 186.07931,
    "U": 150.95364,
    "O": 237.14773,
}
_PROTON = 1.007276466812
_WATER = 18.0105646863

#: Captures a residue and an optional following modification group, e.g. ``C(UniMod:4)`` or ``M[+16]``.
_ANNOT_RESIDUE_PATTERN = _re.compile(
    r"([A-Za-z])(?:\(([^)]*)\)|\[([^\]]*)\])?"
)


def _mod_delta_lookup() -> _Dict[str, float]:
    """Map a modification name (e.g. ``UniMod:4``) to its monoisotopic delta, reusing the heuristic table."""
    return {name: delta for name, delta in _get_mod_heuristic_tbl()}


def _residue_masses(mod_peptide: str, mod_lookup: _Dict[str, float]) -> list:
    """Parse a (normalized) modified-peptide string into a list of per-residue monoisotopic masses."""
    masses = []
    for m in _ANNOT_RESIDUE_PATTERN.finditer(mod_peptide.strip("_")):
        aa = m.group(1).upper()
        if aa not in _AA_RESIDUE_MASS:
            continue
        mass = _AA_RESIDUE_MASS[aa]
        mod = m.group(2) or m.group(3)
        if mod:
            if mod in mod_lookup:
                mass += mod_lookup[mod]
            else:
                try:
                    mass += float(mod)
                except ValueError:
                    pass  # unknown mod; best-effort, may fail to match and be dropped
        masses.append(mass)
    return masses


def _theoretical_by_ions(residue_masses: list, max_charge: int) -> list:
    """Return ``(mz, ion_type, series_number, charge)`` for b/y ions at charges ``1..max_charge``."""
    ions = []
    n = len(residue_masses)
    cumulative = 0.0
    for i in range(n - 1):  # b ions (N-terminal)
        cumulative += residue_masses[i]
        for z in range(1, max_charge + 1):
            ions.append(((cumulative + z * _PROTON) / z, "b", i + 1, z))
    cumulative = _WATER
    for i in range(n - 1):  # y ions (C-terminal)
        cumulative += residue_masses[n - 1 - i]
        for z in range(1, max_charge + 1):
            ions.append(((cumulative + z * _PROTON) / z, "y", i + 1, z))
    return ions


def _annotate_fragment_ions(
    df: "_pd.DataFrame", max_charge: int = 2, tol: float = 0.05
) -> "_pd.DataFrame":
    """
    Add ``FragmentType``, ``FragmentCharge``, and ``FragmentSeriesNumber`` columns to a library frame.

    For each precursor the theoretical b/y ion ladder is computed at charges ``1..max_charge`` and each
    ``ProductMz`` is matched to its nearest theoretical ion within ``tol`` Da. Fragments that match no b/y
    ion are dropped: they cannot be assigned a valid ion identity, and a consumer that re-derives ion
    identities would reject them.

    Why this matters: the library TSV is otherwise written with only ``ProductMz``/``LibraryIntensity`` at
    the fragment level. A consumer re-deriving ion identities from a *charge-1-only* theoretical ladder (as
    the Radiant engine does when it re-reads the MBR library for the second-pass search) cannot place
    **multiply-charged fragment ions** -- which modern predictors (e.g. DIA-NN 2.x) emit in quantity -- and
    aborts on the first one. Emitting ``FragmentCharge`` (with type/series) lets the consumer use the ion
    identity directly instead of re-deriving it.
    """
    if (
        df.empty
        or "ModifiedPeptide" not in df.columns
        or "ProductMz" not in df.columns
    ):
        return df

    mod_lookup = _mod_delta_lookup()
    product_mz = df["ProductMz"].to_numpy(dtype=float)
    frag_type = [None] * len(df)
    frag_charge = [0] * len(df)
    frag_series = [-1] * len(df)

    # `.indices` gives integer *positional* arrays per group, aligning with `product_mz`.
    for peptide, positions in df.groupby(
        "ModifiedPeptide", sort=False
    ).indices.items():
        ions = _theoretical_by_ions(
            _residue_masses(str(peptide), mod_lookup), max_charge
        )
        if not ions:
            continue
        for pos in positions:
            mz = product_mz[pos]
            best = None
            best_d = tol
            for imz, itype, iseries, icharge in ions:
                d = abs(imz - mz)
                if d < best_d:
                    best_d = d
                    best = (itype, iseries, icharge)
            if best is not None:
                frag_type[pos], frag_series[pos], frag_charge[pos] = best

    result = df.copy()
    result["FragmentType"] = frag_type
    result["FragmentCharge"] = frag_charge
    result["FragmentSeriesNumber"] = frag_series

    matched = result["FragmentType"].notna()
    n_drop = int((~matched).sum())
    if n_drop:
        _logger.warning(
            "Fragment annotation: dropped %d of %d fragment(s) matching no b/y ion within %.3g Da",
            n_drop,
            len(result),
            tol,
        )
    return result.loc[matched].reset_index(drop=True)


def write_library(
    peptides: _PsmDataset,
    proteins: _Optional[_Any] = None,
    location: _Optional[str] = None,
    spectra_backend: _Union[str, _Callable] = None,
    threshold_col: _Optional[_Union[str, _Column]] = None,
    qval_thresh: float = None,
    include_decoys: bool = False,
    peptide_normalizer: _Optional[_Dict[str, _Any]] = None,
    output_location: _Optional[str] = None,
    use_dbfs_for_s3: _Optional[bool] = None,
    peptide_kwargs: _Optional[dict] = None,
    protein_kwargs: _Optional[dict] = None,
    annotate_fragment_ions: bool = True,
    fragment_annotation_max_charge: int = 2,
    fragment_annotation_tol: float = 0.05,
    **kwargs,
) -> _DataFrame:
    """
    Write the given dataset to the given location, formatted for use as a spectral library.

    **Filtering** -- The ``qval_thresh`` and ``include_decoys`` parameters allow convenient filtering of output
    PSMs or proteins.

    For more sophisticated filtering, the optional ``threshold_col`` parameter includes only rows where this column
    is ``True`` in the output. When ``threshold_col`` is specified the ``qval_thresh`` and ``include_decoys`` parameters will be ignored.


    **Spectral information** -- This module is meant to consume scored and filtered sets of PSMs, that
    may not necessarily include the necessary spectral information for creating a library.
    Retrieval of this information is implemented by a pluggable backend implementation from `wheely-mammoth <https://github.com/seerbio/wheely-mammoth/blob/main/wheely/mammoth/spectra/parsers/registry.py>`_ capable of
    fetching the precursor- and fragment-level spectral information for PSMs in a filtered dataset.
    This is not required if ``dataset`` implements :py:class:`wheely.mammoth.spectra.SpectraDataset`.

    **Output** -- Libraries are written in a TSV format compatible with DIA-NN and EncyclopeDIA, and
    suitable for conversion to other formats using existing tools. For more information see
    `DIA-NN format documentation <https://github.com/vdemichev/DiaNN#spectral-library-formats>`_.
    Each row represents a single fragment ion in the library. If ``location`` is truthy
    the library will be written to that location. In all cases, the same dataset is returned by
    this function as a PySpark DataFrame.

    Specifically, the following columns are included, in order:

    These columns are the same for each ion in an entry:

    - ``ModifiedPeptide`` -- a string representation of the peptide and modifications. This will be
        taken from the input dataset's ``peptide_column`` then (optionally) normalized by the
        specified ``peptide_normalizer``.
    - ``PrecursorCharge``
    - ``PrecursorMz``
    - ``Tr_recalibrated`` -- The retention time of the ID in an arbitrary scale (possibly all the same
        value, always numeric)
    - ``decoy`` -- a boolean column indicating whether the PSM is a decoy. Note that decoys are only included in the
        output when ``include_decoys`` is ``True`` or when ``threshold_col`` is specified and includes decoys.

    These columns are specific to each ion in an entry:

    - ``ProductMz``
    - ``LibraryIntensity`` -- relative intensity of the fragment; guaranteed to be numeric and non-negative

    Additional columns that will be written conditionally:

    - ``FragmentType``, ``FragmentCharge``, ``FragmentSeriesNumber`` -- the b/y ion identity of each fragment,
        written when ``annotate_fragment_ions`` is true (the default). These let a consumer that re-reads the
        library use the ion identity directly rather than re-deriving it; without ``FragmentCharge`` in
        particular, a consumer that re-derives a charge-1-only ladder cannot place multiply-charged fragment
        ions. See ``annotate_fragment_ions`` below.
    - ``QValue`` -- *q*-value if the dataset is a :py:class:`wheely.mammoth.ConfidenceDataset`
    - ``IonMobility`` -- currently never written

    Currently column names can not be controlled, and are the same regardless of the input dataset
    and its column names, unless noted above.

    Future directions:

    * Add support for other dimensions: e.g. IM
    * Add support for customizing output: e.g. column names, Spark output kwargs, etc.

    Parameters
    ----------
    peptides : PsmDataset
        The dataset
    proteins :
        Ignored
    location : str
        The output location (path or URI)
    output_location.: DEPRECATED
        Synonym for ``location``
    spectra_backend : str | callable
        The backend implementation used to look up library spectral
        information for each supplied PSM.
    threshold_col : str | pyspark.sql.Column; optional
        A column (or its name) specifying which
        rows will be included in the resulting library.
    qval_thresh : float; default = 0.01
        The largest *q*-value accepted into the library. Ignored if
        the dataset is not a :py:class:`wheely.mammoth.ConfidenceDataset` or ``threshold_col`` is specified.
    include_decoys : bool; default = False
        If true, include decoy PSMs in the library. Ignored if ``threshold_col`` is specified.
    peptide_normalizer : dict
        A dict whose ``backend`` (a ``Callable``) will be called to
        normalize each ``ModifiedPeptide`` value (from ``dataset.peptide_column``).

        Any dict entries other than ``backend`` will be passed to the callable as keyword arguments.

        If unspecified or ``None`` a generic normalizer will be used, which provides a "best-effort" normalization
        to DIA-NN like Unimod format (*e.g.* ``C(Unimod:4)``) (see :py:func:`.normalize_peptide_heuristic`).

        A false-y value for ``peptide_normalizer``
        or ``peptide_normalizer["backend"]`` will disable normalization.

        TODO: support a registry of available backend normalizers and permit ``backend`` to be a str
    peptide_kwargs : dict
        Keyword arguments; if ``threshold_col``, ``qval_thresh``, or ``include_decoys`` are not specified, they can
        be included in this dict.
        Other entries will be merged with ``kwargs`` and passed to the ``spectra_backend``.
    protein_kwargs : dict
        Ignored.
    annotate_fragment_ions : bool; default = True
        If true, add ``FragmentType``/``FragmentCharge``/``FragmentSeriesNumber`` to the written library by
        matching each ``ProductMz`` to the theoretical b/y ion ladder (at charges up to
        ``fragment_annotation_max_charge``). Fragments matching no b/y ion within
        ``fragment_annotation_tol`` are dropped. Only the written file is affected; the returned DataFrame is
        unchanged.
    fragment_annotation_max_charge : int; default = 2
        Highest fragment charge state considered when annotating fragment ions.
    fragment_annotation_tol : float; default = 0.05
        Match tolerance (Da) between a ``ProductMz`` and a theoretical b/y ion when annotating.
    kwargs :
        Any additional keyword arguments are passed to ``spectra_backend``.

    Returns
    -------
    out : pyspark.sql.DataFrame
        A PySpark DataFrame with the same contents as the output library.
    """
    if not location:
        location = output_location

    _spectra_backend: _Callable
    if not isinstance(peptides, _SpectraDataset):
        # Fail-fast if a spectral backend is required but not provided
        if not spectra_backend:
            raise ValueError("spectra_backend may not be None!")

        if callable(spectra_backend):
            _spectra_backend = spectra_backend
        else:
            _spectra_backend = _get_spectra_backend(spectra_backend)

    if peptide_kwargs is None:
        peptide_kwargs = {}
    if threshold_col is None:
        threshold_col = peptide_kwargs.pop("threshold_col", None)
    if qval_thresh is None:
        qval_thresh = peptide_kwargs.pop("qval_thresh", 0.01)
    if include_decoys is None:
        include_decoys = peptide_kwargs.pop("include_decoys", False)

    # 1. Filter / normalize
    psms = filter_psms(peptides, threshold_col, qval_thresh, include_decoys)

    if _logger.isEnabledFor(_logging.INFO):
        n_filt = psms.data.count()
        _logger.info("Building library from %d PSMs (after filtering)", n_filt)
        assert n_filt >= 0
    else:
        assert not psms.data.isEmpty()

    if peptide_normalizer is None:
        _logger.debug(
            "Normalizing peptide sequences and mods with default normalizer"
        )
        norm_psms = _normalize_peptides(psms)
    elif peptide_normalizer:
        _logger.debug(
            "Normalizing peptide sequences and mods with: %s",
            peptide_normalizer,
        )
        norm_psms = _normalize_peptides(psms, **peptide_normalizer)
    else:
        _logger.debug("Skipping normalization of peptide sequences and mods")
        norm_psms = psms

    # 2. Join spectral info (if necessary)
    joined_df: _DataFrame
    if isinstance(peptides, _SpectraDataset):
        assert isinstance(
            norm_psms, _SpectraDataset
        ), "Normalized dataset is no longer a SpectraDataset!!"

        joined_df = norm_psms.data

        peptide_col = norm_psms.peptide_column
        target_col = norm_psms.target_column

        charge_col = norm_psms.charge_column
        mz_col = norm_psms.mz_column
        rt_col = norm_psms.rt_column
        peaklist_col = norm_psms.peaklist_column

        if _logger.isEnabledFor(_logging.INFO):
            n_rows = joined_df.count()
            _logger.info("Will write %d entries to library", n_rows)
            assert n_rows > 0
    else:
        spectra: _SpectraDataset = _spectra_backend(
            norm_psms, **peptide_kwargs, **kwargs
        )

        if _logger.isEnabledFor(_logging.INFO):
            n_spec = spectra.data.count()
            _logger.info("Found %d spectra", n_spec)
            assert n_spec > 0
        else:
            assert not spectra.data.isEmpty()

        assert (
            peptides.spectrum_columns == spectra.spectrum_columns
        ), f"Unsupported: differing spectrum IDs! PSMs had {peptides.spectrum_columns} but spectra had {spectra.spectrum_columns}"

        joined_df = norm_psms.data.alias("psms").join(
            spectra.data.alias("spectra"), on=peptides.spectrum_columns
        )

        peptide_col = f"psms.{norm_psms.peptide_column}"
        target_col = f"psms.{norm_psms.target_column}"

        charge_col = f"spectra.{spectra.charge_column}"
        mz_col = f"spectra.{spectra.mz_column}"
        rt_col = f"spectra.{spectra.rt_column}"
        peaklist_col = f"spectra.{spectra.peaklist_column}"

        if _logger.isEnabledFor(_logging.INFO):
            n_rows = joined_df.count()
            _logger.info(
                "Will write %d entries to library (after join)", n_rows
            )
            assert n_rows > 0

    # Selecting this "explodes" the peaklist into one row per fragment peak
    peak = _peaklist_to_pairs(_fns.col(peaklist_col)).alias("__peak")

    # 3. Build, name, and select columns
    output = (
        joined_df.select(
            # TODO: clarify / document this use of `peptide_column`
            _fns.col(peptide_col).alias("ModifiedPeptide"),
            _fns.col(charge_col).cast("integer").alias("PrecursorCharge"),
            _fns.col(mz_col).alias("PrecursorMz"),
            _fns.col(rt_col).alias("Tr_recalibrated"),
            (~_fns.col(target_col).cast("boolean")).alias("decoy"),
            # We must select this up front, it will be aliased into the correct position below
            *(
                [_fns.col("psms." + peptides.qvalue_column).alias("__qvalue")]
                if isinstance(peptides, _ConfidenceDataset)
                else []
            ),
            peak,
        )
        .select(
            "*",
            _fns.col("__peak").getItem(0).alias("ProductMz"),
            _fns.col("__peak").getItem(1).alias("LibraryIntensity"),
        )
        .drop("__peak")
    )

    # Conditionally append column
    if isinstance(peptides, _ConfidenceDataset):
        output = output.withColumn(
            # Note: We take col name from _dataset_, so we only assume the column is present
            # just in case _filter_psms / with_data returns a different type of dataset.
            "QValue",
            _fns.col("__qvalue"),
        ).drop("__qvalue")

    # 4. Write output
    if location:
        if use_dbfs_for_s3:
            _orig_loc = location
            if location.startswith("s3://"):
                location = f"/dbfs/mnt/{location[len('s3://'):]}"
            elif location.startswith("s3a://"):
                location = f"/dbfs/mnt/{location[len('s3a://'):]}"
            else:
                _logger.warning(
                    "use_dbfs_for_s3: Provided location can't be transformed to DBFS! (%s)",
                    _orig_loc,
                )

            _logger.info(
                "use_dbfs_for_s3: Will write library to %s (for %s)",
                location,
                _orig_loc,
            )

        if location.startswith("dbfs:"):
            # Normalize DBFS URL for use with Pandas
            _orig_loc = location
            location = f"/dbfs{location[len('dbfs:'):]}"
            _logger.info(
                "Will write library to %s (for %s)", location, _orig_loc
            )

        # Cast to avoid warning from mypy
        df: _pd.DataFrame = _cast(_pd.DataFrame, output.toPandas())

        # Annotate fragment ions (FragmentType/FragmentCharge/FragmentSeriesNumber) so consumers that
        # re-read this library do not have to re-derive ion identities -- in particular so multiply-charged
        # fragments survive the round-trip. See _annotate_fragment_ions.
        if annotate_fragment_ions:
            df = _annotate_fragment_ions(
                df,
                max_charge=fragment_annotation_max_charge,
                tol=fragment_annotation_tol,
            )

        with _fsspec.open(location, "w") as out:
            df.to_csv(
                out,
                sep="\t",
                header=True,
                index=False,
                quoting=_csv.QUOTE_NONE,
            )

    # 5. Return
    return output


_mod_heuristic_tbl = None


def _get_mod_heuristic_tbl():
    global _mod_heuristic_tbl
    if _mod_heuristic_tbl is None:
        _mod_heuristic_tbl_pattern = _re.compile(
            r"\s*MOD\(\"([^\"]+)\",\s?(?:\(float\)\s*)?(\d*(?:\.\d*)?)\),?"
        )  # TODO
        _mod_heuristic_tbl = [
            (m.group(1), float(m.group(2)))
            for row in r"""
            MOD("UniMod:4", (float)57.021464),
            MOD("Carbamidomethyl (C)", (float)57.021464),
            MOD("Carbamidomethyl", (float)57.021464),
            MOD("CAM", (float)57.021464),
            MOD("+57", (float)57.021464),
            MOD("+57.0", (float)57.021464),
            MOD("UniMod:26", (float)39.994915),
            MOD("PCm", (float)39.994915),
            MOD("UniMod:5", (float)43.005814),
            MOD("Carbamylation (KR)", (float)43.005814),
            MOD("+43", (float)43.005814),
            MOD("+43.0", (float)43.005814),
            MOD("CRM", (float)43.005814),
            MOD("UniMod:7", (float)0.984016),
            MOD("Deamidation (NQ)", (float)0.984016),
            MOD("Deamidation", (float)0.984016),
            MOD("Dea", (float)0.984016),
            MOD("+1", (float)0.984016),
            MOD("+1.0", (float)0.984016),
            MOD("UniMod:35", (float)15.994915),
            MOD("Oxidation (M)", (float)15.994915),
            MOD("Oxidation", (float)15.994915),
            MOD("Oxi", (float)15.994915),
            MOD("+16", (float)15.994915),
            MOD("+16.0", (float)15.994915),
            MOD("Oxi", (float)15.994915),
            MOD("UniMod:1", (float)42.010565),
            MOD("Acetyl (Protein N-term)", (float)42.010565),
            MOD("+42", (float)42.010565),
            MOD("+42.0", (float)42.010565),
            MOD("UniMod:255", (float)28.0313),
            MOD("AAR", (float)28.0313),
            MOD("UniMod:254", (float)26.01565),
            MOD("AAS", (float)26.01565),
            MOD("UniMod:122", (float)27.994915),
            MOD("Frm", (float)27.994915),
            MOD("UniMod:1301", (float)128.094963),
            MOD("+1K", (float)128.094963),
            MOD("UniMod:1288", (float)156.101111),
            MOD("+1R", (float)156.101111),
            MOD("UniMod:27", (float)-18.010565),
            MOD("PGE", (float)-18.010565),
            MOD("UniMod:28", (float)-17.026549),
            MOD("PGQ", (float)-17.026549),
            MOD("UniMod:526", (float)-48.003371),
            MOD("DTM", (float)-48.003371),
            MOD("UniMod:325", (float)31.989829),
            MOD("2Ox", (float)31.989829),
            MOD("UniMod:342", (float)15.010899),
            MOD("Amn", (float)15.010899),
            MOD("UniMod:1290", (float)114.042927),
            MOD("2CM", (float)114.042927),
            MOD("UniMod:359", (float)13.979265),
            MOD("PGP", (float)13.979265),
            MOD("UniMod:30", (float)21.981943),
            MOD("NaX", (float)21.981943),
            MOD("UniMod:401", (float)-2.015650),
            MOD("-2H", (float)-2.015650),
            MOD("UniMod:528", (float)14.999666),
            MOD("MDe", (float)14.999666),
            MOD("UniMod:385", (float)-17.026549),
            MOD("dAm", (float)-17.026549),
            MOD("UniMod:23", (float)-18.010565),
            MOD("Dhy", (float)-18.010565),
            MOD("UniMod:129", (float)125.896648),
            MOD("Iod", (float)125.896648),
            MOD("Phosphorylation (ST)", (float)79.966331),
            MOD("UniMod:21", (float)79.966331),
            MOD("+80", (float)79.966331),
            MOD("+80.0", (float)79.966331),
            MOD("UniMod:259", (float)8.014199, 1),
            MOD("Lys8", (float)8.014199, 1),
            MOD("UniMod:267", (float)10.008269, 1),
            MOD("Arg10", (float)10.008269, 1),
            MOD("UniMod:268", (float)6.013809, 1),
            MOD("UniMod:269", (float)10.027228, 1)
            """.splitlines(
                keepends=False
            )
            if (m := _mod_heuristic_tbl_pattern.match(row))
        ]
    return _mod_heuristic_tbl


#: Pattern used to find modifications that will be string-substituted
_mod_heuristic_pattern = _re.compile(r"([A-Z])(\[.+?\]|\(.+?\))")


def _normalize_mod_heuristic(match: _re.Match) -> str:
    """
    Default "best-effort" modification normalizer, based on heuristics that address only common use
    cases.

    Parameters
    ----------
    match: A match object, corresponding to the ``_mod_heuristic_pattern``.

    Returns
    -------
    A reformatted string meant to be compatible (but not guaranteed to be!) with DIA-NN.
    """
    residue = match.group(1)

    # Ignore captured brackets
    mod = match.group(2)[1:-1]

    try:
        delta = float(mod)
    except:
        # Not a numeric mod; give up!
        pass
    else:
        _tbl = _get_mod_heuristic_tbl()

        closest = min(_tbl, key=lambda p: abs(delta - p[1]))
        if round(delta) == round(closest[1]):
            _logger.debug("Matched mod mass %s to %s", mod, closest)
            return f"{residue}({closest[0]})"

        # Heuristic lookup TODO: likely redundant
        if residue.upper() == "C" and round(delta) == 57:
            return residue + "(UniMod:4)"
        if residue.upper() == "M" and round(delta) == 16:
            return residue + "(UniMod:35)"

    # Give up; return the originally-captured (sub)string
    return match.group(0)


def normalize_peptide_heuristic(seq):
    """
    Default "best-effort" peptide normalizer, based on heuristics that address only common use cases.

    Parameters
    ----------
    seq: A peptide sequence string, including mods in a "typical" format.

    Returns
    -------
    A reformatted string meant to be compatible (but not guaranteed to be!) with DIA-NN.
    """
    return _mod_heuristic_pattern.sub(
        string=seq, repl=_normalize_mod_heuristic
    )


def _normalize_peptides(
    psms: _PsmDataset, backend: _Optional[_Callable] = None, **kwargs
) -> _PsmDataset:
    """
    Normalize each value from ``dataset.peptide_column``.

    peptide_normalizer (dict; optional): A dict whose ``backend`` (a ``callable``) will be called to
        Any dict entries other than ``backend`` will be passed to the callable as keyword arguments.

    Parameters
    ----------
    psms: The dataset that will be normalized
    backend: A callable that will be passed each peptide value, returning the normalized value.
        If unspecified or ``None`` a generic normalizer will be used, which provides a "best-effort"
        normalization to DIA-NN like Unimod format (e.g. ``C(Unimod:4)``). Any other false-y value
        will disable normalization.
        TODO: support a registry of available backend normalizers and permit ``backend`` to be a str
    kwargs: Any keyword arguments will be passed to each invocation of ``backend``.

    Returns
    -------
    A PSM dataset with the ``peptide_column`` values normalized by the given backend.
    """
    if backend is None:
        _backend = normalize_peptide_heuristic
    elif not backend:  # type: ignore[truthy-function]
        return psms

    assert callable(_backend)

    assert (
        psms.peptide_column in psms.data.columns
    ), f"Did not find peptide column `{psms.peptide_column}`"

    orig_pep_col = "__peptide_orig"
    return psms.with_data(
        psms.data.withColumnRenamed(psms.peptide_column, orig_pep_col)
        .withColumn(
            psms.peptide_column,
            _fns.udf(lambda seq: _backend(seq, **kwargs))(
                _fns.col(orig_pep_col)
            ),
        )
        .drop(orig_pep_col)
    )
