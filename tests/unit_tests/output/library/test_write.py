import os.path
from typing import Callable

import pandas as pd
import pytest

from pyspark.sql.functions import array, col, lit, rand as _rand

from wheely.mammoth import ConfidenceDataset, PsmDataset
from wheely.mammoth.parsers import read_encyclopedia_features
from wheely.mammoth.spectra import SpectraDataset, PeaklistType

from scry.output.library import write_library
from scry.output.library.write import _filter_psms


def rand(seed=0):
    """
    Force specifying a seed to aid repeatability of tests.
    """
    return _rand(seed=seed)


@pytest.fixture
def confidence_dataset(psm_dataset) -> ConfidenceDataset:
    # Create a ConfidenceDataset fixture
    dataset = ConfidenceDataset(
        psm_dataset.data.withColumn("q-value", rand()),
        qvalue_column="q-value",
        target_column=psm_dataset.target_column,
        spectrum_columns=psm_dataset.spectrum_columns,
        score_columns=psm_dataset.score_columns,
        peptide_column=psm_dataset.peptide_column,
        protein_column=psm_dataset.protein_column,
        protein_delim=psm_dataset.protein_delim,
    )

    assert all(
        dataset.data.groupBy(dataset.targets).count().toPandas()["count"] > 0
    )

    yield dataset


@pytest.fixture
def psm_dataset() -> PsmDataset:
    # Create a PsmDataset fixture
    dataset = read_encyclopedia_features(
        "data/2017dec27_overlap_dia_6b_rep1_604to616.dia.features.txt"
    )
    yield dataset


@pytest.fixture
def spectra_backend() -> Callable:
    """
    Returns
    -------
    A callable spectral backend compatible with the `psm_dataset` and `confidence_dataset` fixtures.
    """

    def get_spectra(psms: PsmDataset) -> SpectraDataset:
        """
        Returns fake spectra for each input PSM.
        """
        return SpectraDataset(
            psms.data.select(
                *psms.spectrum_columns,
            )
            .withColumn("__z", (rand(seed=0) * 4).astype("int"))
            .withColumn("__mz", rand(seed=1) * 600 + 400)
            .withColumn("__rt", col("__mz") + rand(seed=2) * 200)
            .withColumn(
                "__peaklist",
                array(
                    array(lit(1234.567890), lit(0.67)),
                    array(lit(123.456789), lit(0.33)),
                ).astype(PeaklistType),
            ),
            spectrum_columns=psms.spectrum_columns,
            charge_column="__z",
            mz_column="__mz",
            rt_column="__rt",
            peaklist_column="__peaklist",
        )

    return get_spectra


@pytest.mark.parametrize(
    "dataset_fixture", ["psm_dataset", "confidence_dataset"]
)
def test_create_library(request, dataset_fixture, spectra_backend):
    """
    Test that we can build a library DataFrame from our PSM and spectrum fixtures.
    """
    dataset = request.getfixturevalue(dataset_fixture)

    result = write_library(
        dataset, spectra_backend=spectra_backend, output_location=None
    )

    # print(result.toPandas())

    # Check that the result has at least one peak per spectrum
    assert result.count() >= dataset.data.count()

    # Check for expected columns
    for col in [
        "ModifiedPeptide",
        "PrecursorCharge",
        "PrecursorMz",
        "ProductMz",
    ]:
        assert col in result.columns

    if isinstance(dataset, ConfidenceDataset):
        assert "QValue" in result.columns

    for col in result.columns:
        assert not col.startswith("_"), f"Leaked internal column? {col}"


def test_create_library_output(psm_dataset, spectra_backend, tmp_path):
    """
    Test that we can build a library DataFrame from our PSM and spectrum fixtures.
    """
    output_loc = tmp_path / "test.tsv"

    result = write_library(
        psm_dataset,
        spectra_backend=spectra_backend,
        output_location=str(output_loc),
    )

    # print(result.toPandas())

    # Check that the result has at least one peak per spectrum
    assert result.count() >= psm_dataset.data.count()

    # Check for expected columns
    for col in [
        "ModifiedPeptide",
        "PrecursorCharge",
        "PrecursorMz",
        "ProductMz",
    ]:
        assert col in result.columns

    if isinstance(psm_dataset, ConfidenceDataset):
        assert "QValue" in result.columns

    for col in result.columns:
        assert not col.startswith("_"), f"Leaked internal column? {col}"

    assert os.path.exists(output_loc), "Did not find output file!"

    df = pd.read_csv(output_loc, sep="\t")

    assert len(df) == result.count()

    for col in [
        "ModifiedPeptide",
        "PrecursorCharge",
        "PrecursorMz",
        "ProductMz",
    ]:
        assert col in df.columns


def test_filter_psms_with_confidence_dataset(confidence_dataset):
    # Test data
    threshold_col = None
    qval_thresh = 0.01

    # Call the function
    filtered_dataset = _filter_psms(
        confidence_dataset, threshold_col, qval_thresh
    )

    # Perform assertions (e.g., check if the filtered dataset contains the expected PSMs)
    assert confidence_dataset.data.count() > filtered_dataset.data.count()
    assert filtered_dataset.data.filter(
        filtered_dataset.qvalues > qval_thresh
    ).isEmpty()


def test_filter_psms_with_psm_dataset(psm_dataset):
    # Test data
    threshold_col = "threshold"
    qval_thresh = None

    psm_dataset = psm_dataset.with_data(
        psm_dataset.data.withColumn("threshold", rand() >= 0.5)
    )

    # Call the function
    filtered_dataset = _filter_psms(psm_dataset, threshold_col, qval_thresh)

    # Perform assertions (e.g., check if the filtered dataset contains the expected PSMs)
    assert psm_dataset.data.count() > filtered_dataset.data.count()
    assert filtered_dataset.data.filter(~col(threshold_col)).isEmpty()


@pytest.mark.parametrize(
    "dataset_fixture", ["psm_dataset", "confidence_dataset"]
)
def test_filter_psms_no_filtering(request, dataset_fixture):
    """
    Runs on both dataset fixtures to check the same logic applies to both kinds of dataset.
    """
    dataset = request.getfixturevalue(dataset_fixture)

    # Test data
    threshold_col = None
    qval_thresh = None

    # Call the function
    filtered_dataset = _filter_psms(dataset, threshold_col, qval_thresh)

    # Perform assertions (e.g., check if the filtered dataset is the same as the original dataset)
    assert dataset.data.count() == filtered_dataset.data.count()
