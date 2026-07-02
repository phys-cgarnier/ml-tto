import warnings
import traceback
import time
from typing import Callable, Optional, Tuple
import logging

import numpy as np
from pydantic import PositiveFloat, PositiveInt, Field, field_serializer
from xopt import Xopt, Evaluator, VOCS
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.numerical_optimizer import GridOptimizer
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
    retry_if_exception_type,
    Retrying,
    RetryError,
)

from lcls_tools.common.measurements.emittance_measurement import (
    QuadScanEmittance,
    QuadScanEmittanceResult,
)
from lcls_tools.common.measurements.screen_profile import (
    ScreenBeamProfileMeasurement,
)
from ml_tto.automatic_emittance.scan_cropping import crop_scan
from ml_tto.automatic_emittance.transmission import TransmissionMeasurement
from ml_tto.gpsr.lcls_tools import (
    get_lcls_tools_data,
    process_automatic_emittance_measurement_data,
)
from ml_tto.gpsr.quadrupole_scan_fitting import gpsr_fit_quad_scan
from ml_tto.errors import NotReadyError, BackgroundMismatchError

logger = logging.getLogger("auto_quad_scan")


class MLQuadScanEmittance(QuadScanEmittance):
    """
    Machine learning-based quadrupole scan emittance measurement.

    This class uses Bayesian optimization to explore the quadrupole strength
    and measure the beam size at different quadrupole settings. It uses the
    Xopt library to perform the optimization.

    Attributes
    ----------
    scan_values : list[float]
        List of quadrupole strengths used in the scan.
    n_initial_points : PositiveInt
        Number of initial points for the optimization.
    n_iterations : PositiveInt
        Number of iterations for the optimization.
    max_scan_range : Optional[list[float]]
        Maximum scan range for the quadrupole strength.
    X : Optional[Xopt]
        Xopt object for Bayesian optimization.
    min_signal_to_noise_ratio : float
        Minimum signal-to-noise ratio for valid measurements.
    n_interpolate_points : Optional[PositiveInt]
        Number of interpolation measurements made in-between BO-chosen points.
    n_grid_points : PositiveInt
        Number of grid points for the numerical optimizer.
    min_beamsize_cutoff : float
        Minimum beam size cutoff in microns.
    beamsize_cutoff_max : float
        Maximum beam size cutoff as a multiple of the minimum beam size measured.
    beta : float
        Exploration parameter for the Bayesian optimization.
    visualize_bo : bool
        Whether to visualize the Bayesian optimization process.
    visualize_cropping : bool
        Whether to visualize the cropping of the scan.
    verbose : bool
        Whether to print verbose output during the measurement.
    evaluate_callback : Optional[callable]
        Optional callback function to evaluate additional metrics at each quad
        strength during the scan. Should be in the form of
        `evaluate_callback(inputs: dict, fit_result: ImageProjectionFitResult) -> dict`.
        Additional results will be added to the `X.data` attribute.

    """

    beamsize_measurement: ScreenBeamProfileMeasurement
    save_location: Optional[str] = None

    # basic settings for the scan
    n_initial_points: PositiveInt = 5
    n_iterations: PositiveInt = 5
    max_scan_range: Optional[list[float]] = [-10.0, 10.0]

    # visualization settings
    visualize_bo: bool = False
    visualize_cropping: bool = False
    verbose: bool = False

    # more detailed settings for the scan
    n_interpolate_points: Optional[PositiveInt] = 3
    n_grid_points: PositiveInt = 100
    min_beamsize_cutoff: float = 100.0  # in microns
    beamsize_cutoff_max: float = 3.0
    beta: float = 10000.0
    evaluate_callback: Optional[Callable] = Field(None, exclude=True)
    ready_callback: Optional[Callable] = Field(None, exclude=True)
    transmission_measurement: Optional[TransmissionMeasurement] = None
    transmission_measurement_constraint: Optional[float] = 0.9
    max_measurement_retries: int = 10
    bctrl_refresh_rate: float = 0.02

    # data storage
    X: Optional[Xopt] = None
    scan_values: Optional[list[float]] = []

    @field_serializer("X")
    def serialize_X(self, X, info):
        info = X.model_dump() if X is not None else None
        if info is not None:
            # remove the generator's model to avoid serialization issues
            info["generator"].pop("model", None)

            # remove the evaluator's function to avoid serialization issues
            info["evaluator"].pop("function", None)

        return info

    # retry the ready check for at most 5 minutes - check every 5 seconds
    @retry(
        stop=stop_after_delay(300),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(NotReadyError),
    )
    def ready_check(self):
        if self.ready_callback is not None:
            # check to see if we are ready -- if not raise an error
            logger.debug("calling ready check")
            if not self.ready_callback():
                raise NotReadyError()

    def _evaluate(self, inputs):
        # validate that we are ready to set the quad
        self.ready_check()

        # set quadrupole strength
        logger.debug(f"Setting quadrupole strength to {inputs['k']}")
        self.magnet.bctrl = inputs["k"]

        # start by waiting one refresh cycle for bctrl
        # then wait for bact to match bctrl
        # bctrl refresh rate is less than 10 ms
        start_time = time.time()
        ii = 0
        while not np.isclose(self.magnet.bact, inputs["k"], atol=0.01):
            time.sleep(self.bctrl_refresh_rate)
            ii += 1
            if ii % 100 == 0: # log every 100 cycles (every 2 seconds if refresh rate is 20 ms)
                logger.debug(
                    f"Waiting for quadrupole strength to update... set val: {inputs['k']}, bact: {self.magnet.bact}"
                )

        logger.debug(
            f"Quadrupole strength bact is {self.magnet.bact:.4f}, took {time.time() - start_time:.2f} seconds to update"
        )

        # if provided, check transmission
        if self.transmission_measurement is not None:
            transmission = self.transmission_measurement.measure()["transmission"]

            # if the transmission is below the constraint, reset the quad value
            if transmission < self.transmission_measurement_constraint:
                logger.warning(
                    f"transmission {transmission} is below constraint: "
                    f"{self.transmission_measurement_constraint}"
                )
                results = {
                    "x_rms_micron_sq": np.array([np.nan]),
                    "y_rms_micron_sq": np.array([np.nan]),
                    "min_signal_to_noise_ratio": np.nan,
                    "transmission": transmission,
                }
                return results

        self.ready_check()

        # do the measurement of the beam size and callbacks
        logger.debug("trying beamsize measurement")

        self.measure_beamsize()

        fit_result = self._info[-1]
        self.scan_values.append(inputs["k"])

        # if transmission measurement is set, measure transmission
        extra_measurements = {}
        if self.transmission_measurement is not None:
            extra_measurements.update(self.transmission_measurement.measure())

        if self.evaluate_callback is not None:
            additional_results = self.evaluate_callback(
                inputs=inputs, fit_result=fit_result
            )
            extra_measurements.update(additional_results)

        # add extra measurements to fit result metadata
        fit_result.metadata.update(extra_measurements)

        self._info[-1] = fit_result

        # collect results
        rms_x = fit_result.rms_sizes_all[:, 0]
        rms_y = fit_result.rms_sizes_all[:, 1]

        results = {
            "x_rms_micron_sq": rms_x**2,
            "y_rms_micron_sq": rms_y**2,
            "min_signal_to_noise_ratio": np.min(fit_result.signal_to_noise_ratios),
        }
        results.update(extra_measurements)

        return results

    def create_xopt_object(self, vocs):
        evaluator = Evaluator(function=self._evaluate)
        generator = UpperConfidenceBoundGenerator(
            vocs=vocs,
            beta=self.beta,
            numerical_optimizer=GridOptimizer(n_grid_points=self.n_grid_points),
            n_interpolate_points=self.n_interpolate_points,
            n_monte_carlo_samples=64,
        )
        self.X = Xopt(vocs=vocs, evaluator=evaluator, generator=generator)

    def update_xopt_object(self, vocs):
        self.X.generator.vocs = vocs

    def reset(self):
        self.scan_values = []
        self._info = []

    def run_iterations(self, dim_name, n_iterations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(n_iterations):
                logger.info(f"Iteration {i} for dimension {dim_name}")
                self.update_xopt_object(self.get_vocs(dim_name))

                if self.visualize_bo:
                    self.X.generator.train_model()
                    self.X.generator.visualize_model(
                        exponentiate=True,
                        show_feasibility=True,
                    )

                self.X.step()

    def perform_beamsize_measurements(self):
        """
        Run BO-based exploration of the quadrupole strength to get beamsize measurements
        """

        # ignore warnings from UCB generator and Xopt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.create_xopt_object(self.get_vocs("x"))

        # get current value of k
        current_k = self.magnet.bctrl

        # fast scan to get initial guess -- start from current k and scan to the far end of the range
        initial_scan_values = np.linspace(
            current_k,
            self.max_scan_range[0]
            if np.abs(current_k - self.max_scan_range[0])
            > np.abs(current_k - self.max_scan_range[1])
            else self.max_scan_range[1],
            self.n_initial_points,
        )

        try:
            self.X.evaluate_data({"k": initial_scan_values})

            # run iterations for x/y -- ignore warnings from UCB generator
            logger.info("Running x scans")
            self.run_iterations("x", self.n_iterations + 2) # add 2 extra iterations for x
            logger.info("Running y scans")
            self.run_iterations("y", self.n_iterations)

        except Exception as e:
            raise e
        finally:
            # reset quadrupole strength to original value
            self.magnet.bctrl = current_k

    def _get_beamsizes_scan_values_from_info(self) -> Tuple[np.ndarray]:
        """
        Extract the mean rms beam sizes from the info list, units in meters.
        """
        beam_sizes = []
        for ele in self._info:
            beam_sizes.append(ele.rms_sizes.reshape(1, 2) * 1e-6)  # convert to meters

        # get scan values and extend for each direction
        scan_values = np.tile(np.array(self.scan_values), (2, 1))

        beam_sizes = np.vstack(beam_sizes).T

        scan_values_cropped = []
        beam_sizes_cropped = []
        dim_names = ["x", "y"]
        for i in range(2):
            # crop the scans using concavity filter and max beam size filter
            cutoff_size = self._get_cutoff_beamsize(dim_names[i])
            sv_cropped, bs_cropped, _, _, _, _ = crop_scan(
                scan_values=scan_values[i],
                beam_sizes_squared=(beam_sizes[i]*1e6)**2,
                cutoff_max=cutoff_size,
                cutoff_min=1.0,  # minimum beam size of 1 micron
            )

            scan_values_cropped += [sv_cropped]
            beam_sizes_cropped += [bs_cropped**0.5 * 1e-6]  # convert back to meters

        return scan_values_cropped, beam_sizes_cropped

    def get_vocs(self, dim_name):
        """
        Utility function to create x/y vocs.

        This function creates a VOCS object for the given dimension name (x or y).
        It sets the objectives to minimize the rms beam size in pixel squared for that dimension.
        It also sets the constraints based on the minimum signal-to-noise ratio
        and the maximum beam size cutoff based on the smallest beam size measured.

        If a transmission measurement is set, it will also add a transmission constraint to the vocs.

        """

        scan_name = f"{dim_name}_rms_micron_sq"
        vocs = VOCS(
            variables={"k": self.max_scan_range},
            objectives={scan_name: "MINIMIZE"},
            observables=["x_rms_micron_sq", "y_rms_micron_sq"],
        )

        if self.X is not None:
            if self.X.data is not None:
                vocs.constraints = {
                    "min_signal_to_noise_ratio": [
                        "GREATER_THAN",
                        self.beamsize_measurement.beam_fit.signal_to_noise_threshold,
                    ],
                    scan_name: [
                        "LESS_THAN",
                        (self._get_cutoff_beamsize(dim_name)) ** 2,
                    ],
                }

                if self.transmission_measurement is not None:
                    vocs.constraints["transmission"] = [
                        "GREATER_THAN",
                        self.transmission_measurement_constraint,
                    ]

        return vocs

    def _get_cutoff_beamsize(self, dim_name):
        """
        return the cutoff beam size for the given dimension, returned in pixel scale
        """
        param_name = f"{dim_name}_rms_micron_sq"
        min_size = np.nanmin(self.X.data[param_name].to_numpy(dtype="float"))
        return np.max(
            (
                self.beamsize_cutoff_max * np.sqrt(min_size),
                self.min_beamsize_cutoff,
            )
        )


class GPSRMLQuadScanEmittance(MLQuadScanEmittance):
    max_pixels: PositiveInt = 1e5
    n_epochs: PositiveInt = 500
    n_particles: PositiveInt = 10000
    beam_fraction: PositiveFloat = 1.0
    visualize_gpsr: bool = False
    n_stds: PositiveFloat = 5.0
    save_name: str = "gpsr_result"
    image_min_signal_to_noise_ratio: PositiveFloat = 20.0
    median_filter_size: PositiveInt = 3

    def calculate_emittance(self):
        """
        Modify the emittance calculation for the quad scan to utilize GPSR.
        """
        # get the emittance measurement result object
        initial_result = super().calculate_emittance()

        try:
            processed_data = process_automatic_emittance_measurement_data(
                get_lcls_tools_data(initial_result.model_dump()),
                min_signal_to_noise_ratio=self.image_min_signal_to_noise_ratio,
                n_stds=self.n_stds,
                max_pixels=self.max_pixels,
                median_filter_size=self.median_filter_size,
                threshold_multiplier=2.0,
            )

            gpsr_result = gpsr_fit_quad_scan(
                processed_data["quad_strengths"],
                processed_data["images"],
                processed_data["energy"],
                processed_data["rmat"],
                processed_data["resolution"],
                self.n_epochs,
                self.beam_fraction,
                n_particles=self.n_particles,
                design_twiss=processed_data["design_twiss"],
                visualize=self.visualize_gpsr,
                save_location=self.save_location,
                save_name=self.save_name,
            )

            emittance = np.array(
                [
                    gpsr_result["emittance_x"],
                    gpsr_result["emittance_y"],
                ]
            ).reshape(2, 1)
            formatted_result = QuadScanEmittanceResult(
                quadrupole_focusing_strengths=[processed_data["quad_strengths"]] * 2,
                quadrupole_pv_values=[processed_data["quad_pv_values"]] * 2,
                emittance=emittance,
                bmag=gpsr_result["bmag"],
                twiss=gpsr_result["twiss"],
                rms_beamsizes=gpsr_result["rms_beamsizes"],
                beam_matrix=gpsr_result["beam_matrix"],
                metadata=initial_result.metadata,
            )

            return formatted_result, gpsr_result
        except Exception as e:
            print(
                f"Error occurred during GPSR emittance calculation: {traceback.format_exc()} "
                "Returning normal emittance calculation results."
            )
            return initial_result, None
