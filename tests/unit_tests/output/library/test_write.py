from typing import Callable

import pytest

from numpy.testing import assert_array_equal

import pyspark.sql.functions as fns
from pyspark.sql.functions import array, lit

from wheely.mammoth import ConfidenceDataset, PsmDataset
from wheely.mammoth.spectra import (
    SpectraDataset,
    SpectraDatasetBase,
    PeaklistType,
)

from fulcrum.output.library import write_library
from fulcrum.output.library.write import (
    _normalize_peptides,
    _annotate_fragment_ions,
)

from ....conftest import rand


@pytest.fixture
def spectra_dataset(psm_dataset) -> SpectraDataset:
    class PsmSpectraDataset(PsmDataset, SpectraDatasetBase):
        def __init__(
            self,
            psms,
            score_columns,
            spectrum_columns,
            target_column,
            peptide_column,
            protein_column,
            protein_delim,
            charge_column,
            mz_column,
            rt_column,
            peaklist_column,
        ):
            PsmDataset.__init__(
                self,
                psms,
                score_columns=score_columns,
                spectrum_columns=spectrum_columns,
                target_column=target_column,
                peptide_column=peptide_column,
                protein_column=protein_column,
                protein_delim=protein_delim,
            )
            SpectraDatasetBase.__init__(
                self,
                psms,
                spectrum_columns=spectrum_columns,
                charge_column=charge_column,
                mz_column=mz_column,
                rt_column=rt_column,
                peaklist_column=peaklist_column,
            )

        def with_data(self, data, **kwargs):
            return PsmSpectraDataset(
                data,
                **dict(
                    dict(
                        score_columns=self.score_columns,
                        spectrum_columns=self.spectrum_columns,
                        target_column=self.target_column,
                        peptide_column=self.peptide_column,
                        protein_column=self.protein_column,
                        protein_delim=self.protein_delim,
                        charge_column=self.charge_column,
                        mz_column=self.mz_column,
                        rt_column=self.rt_column,
                        peaklist_column=self.peaklist_column,
                    ),
                    **kwargs,
                ),
            )

    dataset = PsmSpectraDataset(
        psm_dataset.data.withColumn("__z", (rand(seed=0) * 4).astype("int"))
        .withColumn("__mz", rand(seed=1) * 600 + 400)
        .withColumn("__rt", fns.col("__mz") + rand(seed=2) * 200)
        .withColumn(
            "__peaklist",
            array(
                array(lit(1234.567890), lit(0.67)),
                array(lit(123.456789), lit(0.33)),
            ).astype(PeaklistType),
        ),
        target_column=psm_dataset.target_column,
        score_columns=psm_dataset.score_columns,
        spectrum_columns=psm_dataset.spectrum_columns,
        peptide_column=psm_dataset.peptide_column,
        protein_column=psm_dataset.protein_column,
        protein_delim=psm_dataset.protein_delim,
        charge_column="__z",
        mz_column="__mz",
        rt_column="__rt",
        peaklist_column="__peaklist",
    )

    assert dataset.peptide_column in dataset.columns
    assert not [c for c in dataset.columns if c not in dataset.data.columns]

    return dataset


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
        return SpectraDatasetBase(
            psms.data.select(
                *psms.spectrum_columns,
            )
            .withColumn("__z", (rand(seed=0) * 4).astype("int"))
            .withColumn("__mz", rand(seed=1) * 600 + 400)
            .withColumn("__rt", fns.col("__mz") + rand(seed=2) * 200)
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
def test_write_library(request, dataset_fixture, spectra_backend):
    """
    Test that we can build a library DataFrame from our PSM and spectrum fixtures.
    """
    dataset = request.getfixturevalue(dataset_fixture)

    result = write_library(
        dataset,
        spectra_backend=spectra_backend,
        output_location=None,
        qval_thresh=1.0,
        include_decoys=True,
        threshold_col=None,
    )

    # print(result.toPandas())

    # Check that the result has at least one peak per spectrum
    assert result.count() >= dataset.data.count()

    # Check for expected columns
    for col in [
        "ModifiedPeptide",
        "PrecursorCharge",
        "PrecursorMz",
        "decoy",
        "ProductMz",
    ]:
        assert col in result.columns

    if isinstance(dataset, ConfidenceDataset):
        assert "QValue" in result.columns

    for col in result.columns:
        assert not col.startswith("__"), f"Leaked internal column? {col}"

    # Assume default sequence / modification normalization
    assert result.filter(fns.col("ModifiedPeptide").rlike(r"\[")).isEmpty()


def _always_throw(*args, **kwargs):
    assert False, "This function should not be invoked!!"


@pytest.fixture(params=[True, False])
def include_decoys(request):
    return request.param


@pytest.mark.parametrize("spectra_backend", [None, _always_throw])
def test_write_library_spectradataset(
    spectra_dataset, spectra_backend, include_decoys
):
    """
    Test that we can build a library DataFrame from our PSM and spectrum fixtures.
    """
    result = write_library(
        spectra_dataset, output_location=None, include_decoys=include_decoys
    )

    # Check that the result has at least one peak per spectrum
    assert result.count() >= spectra_dataset.data.count()

    # Check for expected columns
    for col in [
        "ModifiedPeptide",
        "PrecursorCharge",
        "PrecursorMz",
        "decoy",
        "ProductMz",
    ]:
        assert col in result.columns

    if isinstance(spectra_dataset, ConfidenceDataset):
        assert "QValue" in result.columns

    for col in result.columns:
        assert not col.startswith("__"), f"Leaked internal column? {col}"

    # Assume default sequence / modification normalization
    assert result.filter(fns.col("ModifiedPeptide").rlike(r"\[")).isEmpty()


@pytest.mark.parametrize(
    "dataset_fixture", ["psm_dataset", "confidence_dataset", "spectra_dataset"]
)
def test_normalize_peptides_noop(request, dataset_fixture):
    """
    Runs on both dataset fixtures to check the same logic applies to both kinds of dataset.
    """
    dataset = request.getfixturevalue(dataset_fixture)

    # Call the function
    norm_psms = _normalize_peptides(dataset, backend=False)  # Use no-op

    # Perform assertions (e.g., check if the filtered dataset is the same as the original dataset)
    assert dataset.data.count() == norm_psms.data.count()

    assert_array_equal(
        dataset.data.select(dataset.peptides)
        .toPandas()[dataset.peptide_column]
        .values,
        norm_psms.data.select(norm_psms.peptides)
        .toPandas()[norm_psms.peptide_column]
        .values,
    )


@pytest.mark.parametrize(
    "dataset_fixture", ["psm_dataset", "confidence_dataset", "spectra_dataset"]
)
def test_normalize_peptides_unmod(request, dataset_fixture):
    """
    Check that normalization works for unmodified peptides.
    Runs on both dataset fixtures to check the same logic applies to both kinds of dataset.
    """
    dataset = request.getfixturevalue(dataset_fixture)

    # Filter to just peptides w/o mods
    dataset = dataset.with_data(
        dataset.data.filter(~dataset.peptides.rlike(r"[\[\]\(\)]"))
    )

    assert dataset.data.count() > 0

    # Call the function
    norm_psms = _normalize_peptides(dataset, backend=None)  # Use default

    # Perform assertions (e.g., check if the filtered dataset is the same as the original dataset)
    assert dataset.data.count() == norm_psms.data.count()

    assert_array_equal(
        dataset.data.select(dataset.peptides)
        .toPandas()[dataset.peptide_column]
        .values,
        norm_psms.data.select(norm_psms.peptides)
        .toPandas()[norm_psms.peptide_column]
        .values,
    )


@pytest.mark.parametrize(
    "dataset_fixture", ["psm_dataset", "confidence_dataset", "spectra_dataset"]
)
def test_normalize_peptides_carbamid_metox(request, dataset_fixture):
    """
    Check that normalization works for C+57 and/or M+16.
    Runs on both dataset fixtures to check the same logic applies to both kinds of dataset.
    """
    dataset = request.getfixturevalue(dataset_fixture)

    # Filter to just peptides w/ mods of interest
    dataset = dataset.with_data(
        dataset.data.filter(
            # Assumes EncyclopeDIA-formatted input
            dataset.peptides.rlike(r"C\[(?:\+)?57|M\[(?:\+)?1[56]")
        )
    )

    # Note: we don't check if both mods are present, just that at least one is
    assert dataset.data.count() > 0

    # Call the function
    norm_psms = _normalize_peptides(dataset, backend=None)  # Use default

    # Perform assertions (e.g., check if the filtered dataset is the same as the original dataset)
    assert dataset.data.count() == norm_psms.data.count()

    assert_array_equal(
        dataset.data.select(dataset.peptides)
        .toPandas()[dataset.peptide_column]
        .str.replace(
            r"(?<=C)\[(?:\+)?57(?:\.)?[0-9]*\]", "(UniMod:4)", regex=True
        )
        .str.replace(
            r"(?<=M)\[(?:\+)?1[56](?:\.)?[0-9]*\]", "(UniMod:35)", regex=True
        )
        .values,
        norm_psms.data.select(norm_psms.peptides)
        .toPandas()[norm_psms.peptide_column]
        .values,
    )


def test_annotate_fragment_ions():
    """
    Fragment ions -- including multiply-charged ones -- are annotated with type/charge/series, and
    fragments matching no b/y ion are dropped.

    Regression test for the MBR second-pass crash: without ``FragmentCharge`` a consumer that re-derives a
    charge-1-only ladder cannot place 2+ fragment ions. Values below are real b/y ions of HVVFGHVK from a
    DIA-NN 2.3 predicted library; 339.180 is b6(2+) and 220.634 is y4(2+).
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "ModifiedPeptide": ["HVVFGHVK"] * 6,
            "ProductMz": [
                336.203,
                339.180,
                383.240,
                440.262,
                220.634,
                999.999,
            ],
            "LibraryIntensity": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )

    out = _annotate_fragment_ions(df, max_charge=2, tol=0.05)

    for col in ("FragmentType", "FragmentCharge", "FragmentSeriesNumber"):
        assert col in out.columns

    ann = {
        round(mz, 3): (t, int(c), int(s))
        for mz, t, c, s in zip(
            out["ProductMz"],
            out["FragmentType"],
            out["FragmentCharge"],
            out["FragmentSeriesNumber"],
        )
    }
    assert ann[336.203] == ("b", 1, 3)
    assert ann[339.180] == ("b", 2, 6)  # doubly-charged b ion
    assert ann[383.240] == ("y", 1, 3)
    assert ann[440.262] == ("y", 1, 4)
    assert ann[220.634] == ("y", 2, 4)  # doubly-charged y ion

    # The unmatchable fragment (matches no b/y ion within tolerance) is dropped.
    assert 999.999 not in ann
    assert len(out) == 5
