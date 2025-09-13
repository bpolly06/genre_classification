import pytest
import pandas as pd
import wandb

# Start a W&B run
run = wandb.init(job_type="data_tests")


def pytest_addoption(parser):
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")


@pytest.fixture(scope="session")
def data(request):
    reference_artifact = request.config.option.reference_artifact
    if reference_artifact is None:
        pytest.fail("--reference_artifact missing on command line")

    sample_artifact = request.config.option.sample_artifact
    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")

    # Download artifacts
    ref_path = run.use_artifact(reference_artifact).file()
    sample_path = run.use_artifact(sample_artifact).file()

    sample1 = pd.read_csv(ref_path)
    sample2 = pd.read_csv(sample_path)

    # Columns to test in KS test
    numeric_cols = [
        "danceability","energy","loudness","speechiness",
        "acousticness","instrumentalness","liveness","valence",
        "tempo","duration_ms"
    ]

    # Convert mixed types to numeric and handle NaNs
    sample1[numeric_cols] = sample1[numeric_cols].apply(pd.to_numeric, errors='coerce')
    sample2[numeric_cols] = sample2[numeric_cols].apply(pd.to_numeric, errors='coerce')
    sample1[numeric_cols] = sample1[numeric_cols].fillna(0)
    sample2[numeric_cols] = sample2[numeric_cols].fillna(0)

    return sample1, sample2


@pytest.fixture(scope='session')
def ks_alpha(request):
    ks_alpha = request.config.option.ks_alpha
    if ks_alpha is None:
        pytest.fail("--ks_alpha missing on command line")
    return float(ks_alpha)