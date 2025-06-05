"""
Implementations of median normalization.
"""

from pyspark.sql import (
    Column as _Column,
    functions as _fns,
    Window as _Window,
)

from wheely.mammoth import PsmIntensityDataset as _PsmIntensityDataset

from .base import BasicNormalizer
from .util import get_filtered_intensities as _get_filtered_intensities


class MedianNormalizer(BasicNormalizer):
    """
    Computes median normalization
    """

    def __init__(self):
        self.__name__ = "median"

    def get_normalized_column(
        self,
        dataset: _PsmIntensityDataset,
        *_,
        qval_thresh=None,
        include_decoys=False,
    ) -> _Column:
        """
        Return a :py:class:`~pyspark.sql.Column` that computes median normalization.

        Parameters
        ----------
        dataset : PsmIntensityDataset
            The dataset to normalize.
        qval_thresh : float, optional
            If specified, the median will be computed from only PSMs with *q*-values less than or equal to this value.
        include_decoys : bool, optional
            If ``False`` (default), the median will be computed from only target PSMs.
        """
        if _:
            raise TypeError("Unsupported: additional positional arguments!")

        intensities = _get_filtered_intensities(
            dataset,
            qval_thresh=qval_thresh if qval_thresh is not None else 1.0,
            include_decoys=include_decoys,
        )

        return (
            dataset.intensities
            / _fns.median(intensities).over(
                _Window.partitionBy(dataset.samples)
            )
            # Scale value globally; the exact value is unimportant, so use an efficient estimate
            * _fns.percentile_approx(intensities, 0.5, 1000).over(
                _Window.rowsBetween(
                    _Window.unboundedPreceding, _Window.unboundedFollowing
                )
            )
        )


class MedianDenseNormalizer(MedianNormalizer):
    """
    Computes median normalization
    """

    def __init__(self):
        super().__init__()
        self.__name__ = "mediandense"

    def get_normalized_column(
        self,
        dataset: _PsmIntensityDataset,
        *_,
        qval_thresh=None,
        include_decoys=False,
        density_thresh: float = 0.8,
    ) -> _Column:
        """
        Return a :py:class:`~pyspark.sql.Column` that computes median normalization using only precursors with a density
        above a threshold, as computed by the number of samples with detections.

        Parameters
        ----------
        dataset : PsmIntensityDataset
            The dataset to normalize.
        qval_thresh : float, optional
            If specified, the median will be computed from only PSMs with *q*-values less than or equal to this value.
        include_decoys : bool, optional
            If ``False`` (default), the median will be computed from only target PSMs.
        density_thresh : float, optional
            The density level required for precursors to be used for normalization. Default: 0.8
        """
        if _:
            raise TypeError("Unsupported: additional positional arguments!")

        if getattr(dataset, "charge_column", None) is None:
            raise TypeError("MedianDenseNormalizer requires a charge_column!")

        intensities = _get_filtered_intensities(
            dataset,
            qval_thresh=qval_thresh if qval_thresh is not None else 1.0,
            include_decoys=include_decoys,
        )

        n_samples = (
            dataset.data.select(_fns.countDistinct(dataset.samples))
            .toPandas()
            .iloc[0, 0]
        )

        intensities = _fns.when(
            _fns.countDistinct(dataset.samples).over(
                _Window.partitionBy(dataset.peptides, dataset.charges)
            )
            >= density_thresh * n_samples,
            intensities,
        ).otherwise(_fns.lit(None))

        return (
            dataset.intensities
            / _fns.median(intensities).over(
                _Window.partitionBy(dataset.samples)
            )
            # Scale value globally; the exact value is unimportant, so use an efficient estimate
            * _fns.percentile_approx(intensities, 0.5, 1000).over(
                _Window.rowsBetween(
                    _Window.unboundedPreceding, _Window.unboundedFollowing
                )
            )
        )
