import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from equipo16_accel.refactor.DataProcessor import DataProcessor
from equipo16_accel.refactor.ModelTrainer import ModelTrainer
from equipo16_accel.refactor.Visualizer import Visualizer


def main():
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
    # Create a new Visualizer instance
    l_visualizer = Visualizer()

    # plot dataframe histograms
    l_visualizer.plot_histograms(dataframe, ['x', 'y', 'z'])

    # plot dataframe kde
    l_visualizer.plot_kde(dataframe, ['x', 'y', 'z'])


def prepare_data(dataframe):
    x = dataframe[['x', 'y', 'z', 'pctid', 'vibration_magnitude']]
    y = dataframe['wconfid']
    return x, y


def define_models():
    # Define model list
    return [
        {
            'model': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
            'params': {'n_estimators': 100, 'max_depth': None},
            'name': 'RandomForest_Model',
        },
        {
            'model': SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
            'params': {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
            'name': 'SVM_Model',
        },
        {
            'model': xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, subsample=1.0,
                use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 1.0},
            'name': 'XGBoost_Model',
        }
    ]


def train_and_evaluate_models(models, x_train, x_test, y_train, y_test):
    for model_info in models:
        print(f"Training Model {model_info['name']}")
        trainer = ModelTrainer(
            model=model_info['model'],
            params=model_info['params'],
            model_name=model_info['name'],
            mlflow_experiment="Vibration_Configuration_Classification_refactorizado"
        )

        if model_info['name'] == "SVM_Model":
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            trainer.run_training(x_train_scaled, x_test_scaled, y_train, y_test)
            continue

        trainer.run_training(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
