# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


from __future__ import print_function

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
import os
import sys

from six.moves import urllib

from mlflow.store.file_store import FileStore
from mlflow.store.rest_store import RestStore
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils import env, rest_utils
from mlflow.utils.databricks_utils import get_databricks_host_creds


_TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"
_LOCAL_FS_URI_PREFIX = "file:///"
_REMOTE_URI_PREFIX = "http://"

# Extra environment variables which take precedence for setting the basic/bearer
# auth on http requests.
_TRACKING_USERNAME_ENV_VAR = "MLFLOW_TRACKING_USERNAME"
_TRACKING_PASSWORD_ENV_VAR = "MLFLOW_TRACKING_PASSWORD"
_TRACKING_TOKEN_ENV_VAR = "MLFLOW_TRACKING_TOKEN"
_TRACKING_INSECURE_TLS_ENV_VAR = "MLFLOW_TRACKING_INSECURE_TLS"


_tracking_uri = None


def is_tracking_uri_set():
    """Returns True if the tracking URI has been set, False otherwise."""
    if _tracking_uri or env.get_env(_TRACKING_URI_ENV_VAR):
        return True
    return False


def set_tracking_uri(uri):
    """
    Set the tracking server URI. This does not affect the
    currently active run (if one exists), but takes effect for successive runs.
    :param uri:
                - An empty string, or a local file path, prefixed with ``file:/``. Data is stored
                  locally at the provided file (or ``./mlruns`` if empty).
                - An HTTP URI like ``https://my-tracking-server:5000``.
                - A Databricks workspace, provided as the string "databricks" or, to use a
                  Databricks CLI
                  `profile <https://github.com/databricks/databricks-cli#installation>`_,
                  "databricks://<profileName>".
    """
    global _tracking_uri
    _tracking_uri = uri


def get_tracking_uri():
    """
    Get the current tracking URI. This may not correspond to the tracking URI of
    the currently active run, since the tracking URI can be updated via ``set_tracking_uri``.
    :return: The tracking URI.
    """
    global _tracking_uri
    if _tracking_uri is not None:
        return _tracking_uri
    elif env.get_env(_TRACKING_URI_ENV_VAR) is not None:
        return env.get_env(_TRACKING_URI_ENV_VAR)
    else:
        return os.path.abspath("./mlruns")


def _get_store(store_uri=None):
    store_uri = store_uri if store_uri else get_tracking_uri()
    # Default: if URI hasn't been set, return a FileStore
    if store_uri is None:
        return FileStore()
    # Pattern-match on the URI
    if _is_databricks_uri(store_uri):
        return _get_databricks_rest_store(store_uri)
    if _is_local_uri(store_uri):
        return _get_file_store(store_uri)
    if _is_http_uri(store_uri):
        return _get_rest_store(store_uri)

    raise Exception("Tracking URI must be a local filesystem URI of the form '%s...' or a "
                    "remote URI of the form '%s...'. Update the tracking URI via "
                    "mlflow.set_tracking_uri" % (_LOCAL_FS_URI_PREFIX, _REMOTE_URI_PREFIX))


def _is_local_uri(uri):
    scheme = urllib.parse.urlparse(uri).scheme
    return uri != 'databricks' and (scheme == '' or scheme == 'file')


def _is_http_uri(uri):
    scheme = urllib.parse.urlparse(uri).scheme
    return scheme == 'http' or scheme == 'https'


def _is_databricks_uri(uri):
    """Databricks URIs look like 'databricks' (default profile) or 'databricks://profile'"""
    scheme = urllib.parse.urlparse(uri).scheme
    return scheme == 'databricks' or uri == 'databricks'


def _get_file_store(store_uri):
    path = urllib.parse.urlparse(store_uri).path
    return FileStore(path)


def _get_rest_store(store_uri):
    def get_default_host_creds():
        return rest_utils.MlflowHostCreds(
            host=store_uri,
            username=os.environ.get(_TRACKING_USERNAME_ENV_VAR),
            password=os.environ.get(_TRACKING_PASSWORD_ENV_VAR),
            token=os.environ.get(_TRACKING_TOKEN_ENV_VAR),
            ignore_tls_verification=os.environ.get(_TRACKING_INSECURE_TLS_ENV_VAR) == 'true',
        )
    return RestStore(get_default_host_creds)


def get_db_profile_from_uri(uri):
    """
    Get the Databricks profile specified by the tracking URI (if any), otherwise
    returns None.
    """
    parsed_uri = urllib.parse.urlparse(uri)
    if parsed_uri.scheme == "databricks":
        return parsed_uri.netloc
    return None


def _get_databricks_rest_store(store_uri):
    profile = get_db_profile_from_uri(store_uri)
    return RestStore(lambda: get_databricks_host_creds(profile))


def _get_model_log_dir(model_name, run_id):
    if not run_id:
        raise Exception("Must specify a run_id to get logging directory for a model.")
    store = _get_store()
    run = store.get_run(run_id)
    artifact_repo = ArtifactRepository.from_artifact_uri(run.info.artifact_uri, store)
    return artifact_repo.download_artifacts(model_name)


def _get_git_url_if_present(uri):
    """
    Return the path git_uri#sub_directory if the URI passed is a local path that's part of
    a Git repo, or returns the original URI otherwise.
    :param uri: The expanded uri
    :return: The git_uri#sub_directory if the uri is part of a Git repo,
             otherwise return the original uri
    """
    if '#' in uri:
        # Already a URI in git repo format
        return uri
    try:
        from git import Repo, InvalidGitRepositoryError, GitCommandNotFound, NoSuchPathError
    except ImportError as e:
        print("Notice: failed to import Git (the git executable is probably not on your PATH),"
              " so Git SHA is not available. Error: %s" % e, file=sys.stderr)
        return uri
    try:
        # Check whether this is part of a git repo
        repo = Repo(uri, search_parent_directories=True)

        # Repo url
        repo_url = "file://%s" % repo.working_tree_dir

        # Sub directory
        rlpath = uri.replace(repo.working_tree_dir, '')
        if (rlpath == ''):
            git_path = repo_url
        elif (rlpath[0] == '/'):
            git_path = repo_url + '#' + rlpath[1:]
        else:
            git_path = repo_url + '#' + rlpath
        print("Git path is: %s" % git_path)
        return git_path
    except (InvalidGitRepositoryError, GitCommandNotFound, ValueError, NoSuchPathError):
        return uri


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)


        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
