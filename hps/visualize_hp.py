import argparse
import joblib

import optuna

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualization of study hps results')
    parser.add_argument("path", type=str, help='The path to the study.db to load')
    parser.add_argument("--plot_obj", action="store_true", help='Plot history of the objective')
    
    args = parser.parse_args()
    path_to_study = "sqlite:///{}".format(args.path)

    study_name = path_to_study.split("/")[-1].split(".")[0]
    study = optuna.create_study(study_name=study_name, storage=path_to_study, load_if_exists=True)
    

    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    if args.plot_obj:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
