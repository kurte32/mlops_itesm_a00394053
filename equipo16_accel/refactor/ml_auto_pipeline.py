import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from equipo16_accel.refactor.DataProcessor import DataProcessor
from equipo16_accel.refactor.ModelTrainer import ModelTrainer
from equipo16_accel.refactor.Visualizer import Visualizer


def main():
    """
    Main pipeline function that processes raw data, visualizes it,
    prepares it for training, and trains various machine learning models.
    """

    # get processed data
    processed_data_df = process_raw_data()

    # visualize data
    visualize_data(processed_data_df)

    # prepare data for training, x and y
    x, y = prepare_data(processed_data_df)

    # data splitting into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # create models
    models = define_models()

    # train created models with the training and testing sets
    train_and_evaluate_models(models, x_train, x_test, y_train, y_test)


def process_raw_data():
    """
    Processes the raw data by loading, cleaning, and performing feature engineering.
    Saves the processed data and returns it as a dataframe.
    """

    # Create a new Data processor instance
    data_processor = DataProcessor()

    # load_data takes into account that our dataset is under {project_path}/data/raw
    # if not path provided in DataProcessor object creation
    data_processor.load_data()

    # clean_data removes null values
    data_processor.clean_data()

    # feature engineering
    data_processor.feature_engineering()

    # save processed_data under the selected path or default path otherwise
    data_processor.save_processed_data()

    # returning processed_data dataframe for future usage
    return data_processor.processed_data


def visualize_data(dataframe):
    """
    Visualizes the data using histograms and Kernel Density Estimation (KDE) plots.
    """

    # Create a new Visualizer instance
    l_visualizer = Visualizer()

    # plot dataframe histograms
    l_visualizer.plot_histograms(dataframe, ['x', 'y', 'z'])

    # plot dataframe kde
    l_visualizer.plot_kde(dataframe, ['x', 'y', 'z'])


def prepare_data(dataframe):
    """
    Prepares the features and target variable for model training.
    """
    x = dataframe[['x', 'y', 'z', 'pctid', 'vibration_magnitude']]
    y = dataframe['wconfid']
    return x, y


def define_models():
    """
    Defines the models and their hyperparameter grids for GridSearchCV.
    """
    return [
        {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'max_features': ['sqrt', 'log2', None]
            },
            'name': 'RandomForest_Model',
        },
        {
            'model': SVC(random_state=42),
            'param_grid': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            },
            'name': 'SVM_Model',
        },
        {
            'model': xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            'name': 'XGBoost_Model',
        }
    ]


def get_model(model_name, params):
    """
    Returns the model initialized with the given parameters based on the model name.
    """
    match model_name:
        case 'RandomForest_Model':
            return RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                max_features=params['max_features'],
                random_state=42
            )
        case 'SVM_Model':
            return SVC(
                kernel=params['kernel'],
                C=params['C'],
                gamma=params['gamma'],
                random_state=42
            )
        case 'XGBoost_Model':
            return xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42
            )
        case _:
            raise ValueError(f"Unknown model name: {model_name}")


def train_and_evaluate_models(models, x_train, x_test, y_train, y_test):
    """
    Trains and evaluates each model manually over all parameter combinations
    using ParameterGrid. Logs each model and its performance using MLflow.
    """
    mlflow_experiment = "Vibration_Configuration_Classification_refactorizado"

    for model_info in models:
        # Iterate over all combinations of parameters using ParameterGrid
        print(model_info['name'])
        print(model_info['param_grid'])
        for params in ParameterGrid(model_info['param_grid']):
            # Prepare data, scale if necessary (SVM model)
            if model_info['name'] == 'SVM_Model':
                scaler = StandardScaler()
                x_train_prepared = scaler.fit_transform(x_train)
                x_test_prepared = scaler.transform(x_test)
            else:
                x_train_prepared = x_train
                x_test_prepared = x_test

            # Use get_model to initialize the model with the current parameters
            model = get_model(model_info['name'], params)

            # Train and evaluate the model using ModelTrainer
            trainer = ModelTrainer(model, params, model_info['name'], mlflow_experiment)
            trainer.run_training(x_train_prepared, x_test_prepared, y_train, y_test)

            print(f"Trained {model_info['name']} with params: {params}")


if __name__ == '__main__':
    main()
