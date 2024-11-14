import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import helperC as hlp
import models as mdl
import matplotlib.pyplot as plt
import json
import random
import warnings
warnings.filterwarnings('ignore')

def get_args():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clients', type=int, default=2, help="Number of clients")
    parser.add_argument('--dataset', type=str, default='MNIST', help="Classification dataset")
    parser.add_argument('--model', type=str, default='LeNet5', help="Model architecture")
    parser.add_argument('--alpha', type=float, default=1e15, help="Concentration parameter for data split")
    parser.add_argument('--rounds', type=int, default=100,  help="Number of communication rounds")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini-batch size")
    parser.add_argument('--epoch_per_round', type=int, default=1, help="Number of epoch per communication round")
    parser.add_argument('--lr', type=float, default=1e-3,  help="Learning rate")
    parser.add_argument('--optimizer', type=str, default="adam",  help="Optimizer type")
    parser.add_argument('--feature_dim', type=int, default=100, help="Number of feature")
    parser.add_argument('--n_avg', type=int, default=5, help="Number of considered samples for the averaging")
    parser.add_argument('--lambda_kd', type=float,default=1, help="Meta parameter for feature-based KD")
    parser.add_argument('--lambda_disc', type=float,default=1, help="Meta parameter for contrastive lost")
    parser.add_argument('--kd_type', type=str, default="output",  help="Type of KD (feature or output)")
    parser.add_argument('--sizes',type=float, nargs='*', default=None, help="Dataset sizes")
    parser.add_argument('--reduced', type=float, default=1.0, help="Reduction parameter for the train dataset")
    parser.add_argument('--track_history', type=int, default=1, help="Intervall beween validation evaluation during training")
    parser.add_argument('--fed_avg', type=str, default=False,  help="Use federated averaging for model/classifier/nothing")
    parser.add_argument('--export_dir', type=str, default="./saves", help="Folder to save plots/configuration/etc.")
    parser.add_argument('--data_dir', type=str, default="./data", help="Folder to load/download the data")
    parser.add_argument('--config_file', type=str, default=None, help="Configuration (json) file to reproduce a certain experiment")
    parser.add_argument('--device', type=str, default=None, help="Device to use for learning")
    parser.add_argument('--preset', type=str, default=None, help="Learning preset (fl, fd, il or cl)")
    parser.add_argument('--seed', type=int, default=0, help="Seed for reproductability")
    parser.add_argument('--data_size', type=int, default=90, help="data_size")
    parser.add_argument('--collab_flag', type=bool, default=False, help="collab_flag")
    parser.add_argument('--noise', type=float, default=0.0, help="Noise")
    args = parser.parse_args()
    return args


def run(n_clients=2, dataset="MNIST", model="LeNet5", alpha="uniform", rounds=100, 
        batch_size=32, epoch_per_round=1, lr=1e-4, optimizer="adam", feature_dim=100,
        n_avg=None, lambda_kd=1.0, lambda_disc=1.0, kd_type="output", sizes=None, reduced=False, 
        track_history=1, fed_avg=False, export_dir=None, data_dir="./data", disc_method="classifier",
        device=None, preset=None, seed=0, data_size=90, collab_flag=False,noise=0.0):
    """Run an experiment.
    
    Arguments:
        -n_clients: Number of clients
        -dataset: Classification dataset
        -model: Model architecture
        -alpha: Concentration parameter for data split
        -rounds: Number of communication rounds
        -batch_size: Mini-batch size
        -epoch_per_round: Number of epoch per communication round
        -lr: Learning rate
        -optimizer: Optimizer type
        -feature_dim: Number of feature
        -n_avg: Number of considered samples for the averaging
        -lambda_kd: Meta parameter for feature-based KD
        -lambda_disc: Meta parameter for contrastive lost
        -kd_type: "Type of KD (feature or output)
        -sizes: Dataset sizes
        -reduced: Reduction parameter for the train dataset
        -track_history: Intervall beween validation evaluation during training
        -fed_avg: Use federated averaging for model/classifier/nothing
        -export_dir: Folder to save plots/configuration/etc.
        -data_dir: Folder to load/download the data
        -disc_method: Method for the classifier (classifier or seperate)
        -config_file: Configuration (json) file to reproduce a certain experiment
        -device: Device to use for learning
        -preset: Learning preset (fl, fd, il or cl)
        -seed: Seed for reproductability
    Return:
        - perf_trackers: A list of all the performance trackers
        - feature_trackers: The feature trackers.
    """
    # Argument processing
    if preset is not None:
        if preset in ["fl", "FL", "fedavg", "FedAvg"]:
            print("Running FL with {} clients".format(n_clients))
            lambda_kd=0
            lambda_disc=0
            fed_avg = "model"
        elif preset in ["fd", "FD"]:
            print("Running FD with {} clients".format(n_clients))
            n_avg = None
            lambda_disc=0
            # kd_type = "output"
            fed_avg = False
        elif preset in ["il", "IL"]:
            print("Running IL with {} clients".format(n_clients))
            lambda_kd=0
            lambda_disc=0
            fed_avg = False
        elif preset in ["CL", "cl"]:
            print("Running centralized learning".format(n_clients))
            n_clients = 1
            lambda_kd = 0
            lambda_disc = 0
        else:
            raise ValueError("Unknown preset {} (FL, FD, IL or CL).".format(preset))
    else:
        print("Running CFKD with {} clients".format(n_clients))
    
    if kd_type == "output" and lambda_disc != 0:
        lambda_disc = 0
        print("Warning: lambda_disc has been set to 0 since output-based kd is used (FD).")
    
    # Store parameters for easy reproductability
    if export_dir is not None:
        # Store local parameters
        config = locals()
        
        # Create subdirectories and throw error if already existing (security)
        time_string = time.strftime("%d%h%y_%H:%M:%S")
        directory = os.path.join(export_dir, time_string)
        fig_directory = os.path.join(directory, "figures/")
        os.makedirs(directory, exist_ok=False)
        os.makedirs(fig_directory, exist_ok=False)
        
        # Create config file
        config = json.dumps(config, indent=4)
        with open(os.path.join(directory, 'config.json'), 'w') as outfile:
            outfile.write(config)
            
    # Chose device automatically (if not specified)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print("Device: {}".format(device))
    torch.cuda.empty_cache()
    
    # Reproductilility
    hlp.set_seed(seed)
    
    # set expertise of of clients 
    expertise = [0,0,1,1]
    
    
    partner_client_list = []
    non_partner_client_list = []
    for client_id in range(n_clients):
        p = []
        _np = []
        for exp in range(n_clients):
            if(client_id==exp):
                continue
            if(expertise[client_id] == expertise[exp]):
                p.append(exp)
            else:
                _np.append(exp)
        # print(p)
        # print(np)
        partner_client_list.append(p)
        non_partner_client_list.append(_np)



    # Load data
    train_input, train_target, val_input, val_target, meta = hlp.load_data(dataset=dataset, data_size = data_size, reduced=reduced, device=device)
 
    #Create custom torch datasets
    train_ds = hlp.CustomDataset(train_input, train_target)
    val_ds = hlp.CustomDataset(val_input, val_target)
    
    #Split dataset
            
    # exit() 
    
    train_ds_list, val_ds_list = hlp.split_dataset(n_clients, train_ds, val_ds, alpha, expertise, noise, sizes)
    print(len(train_ds_list))
    print(50*"--")       

    val_ds_list = hlp.uniform_test_split(val_input,val_target,n_clients,meta["n_class"])
    # exit()
    #Create dataloader
    train_dl_list = hlp.ds_to_dl(train_ds_list, batch_size=batch_size)
    val_dl_list = [0]*n_clients
    for _ in range(n_clients):
        val_dl_list[_] = hlp.ds_to_dl(val_ds_list[_], batch_size=batch_size)
    global_val_dl = hlp.ds_to_dl(val_ds, batch_size=batch_size)
    global_train_dl = hlp.ds_to_dl(train_ds, batch_size=batch_size)
    
    #Visualize partition
    hlp.visualize_class_dist(train_ds_list, meta["n_class"], 
                             title="Class distribution (alpha = {})".format(alpha),
                             savepath=os.path.join(fig_directory, "class_dist.png") if export_dir is not None else None)
    
    # Store dataset sizes
    meta["n_total"] = train_input.shape[0] 
    meta["n_local"] = [ds.inputs.shape[0] for ds in train_ds_list]
    
    # Define criterions
    criterion = nn.CrossEntropyLoss()
    criterion_kd = nn.MSELoss(reduction ='none')
    criterion_disc = nn.BCELoss(reduction ='none')
    
    # Model initialization
    client_models = [mdl.get_model(model, feature_dim, meta).to(device) for _ in range(n_clients)]
    if fed_avg in ["model", "classifier"]:
        global_model = mdl.get_model(model, feature_dim, meta).to(device)
    
    # Performance tracker
    if fed_avg == "model":
        perf_trackers = [hlp.PerfTracker(global_model, 
                                         {"Train": train_dl_list[i], "Validation0": val_dl_list[i][0], "Validation1": val_dl_list[i][1], "Validation (global)": global_val_dl}, 
                                         criterion, meta["n_class"], ID="Client {}".format(i)) for i in range(n_clients)]
        
    else: 
      
        perf_trackers = [hlp.PerfTracker(client_models[i], 
                                         {"Train": train_dl_list[i], "Validation0": val_dl_list[i][0], "Validation1": val_dl_list[i][1]}, 
                                         criterion, meta["n_class"], ID="Client {}".format(i)) for i in range(n_clients)]
    
    # parallel trackers for feature and output sharing
    feature_tracker = hlp.OutputTracker([m.features for m in client_models], train_dl_list, feature_dim, meta)
    logits_tracker = hlp.OutputTracker(client_models, train_dl_list, meta["n_class"], meta)
    
    if lambda_disc > 0:
        if disc_method == "classifier":
            discriminators = [mdl.Discriminator("prob_product", client_models[i].classifier) for i in range(n_clients)]
        elif disc_method == "seperate":
            discriminators = [mdl.Discriminator("exponential_prob", feat_dim=feature_dim, n_class=meta["n_class"]).to(device) for i in range(n_clients)]

    # Optimizers
    # Adam
    if optimizer in ["adam", "Adam", "ADAM"]:
        optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in client_models]
        if disc_method == "seperate" and lambda_disc > 0:
            optimizers_disc = [torch.optim.Adam(disc.parameters(), lr=lr) for disc in discriminators]
    
    # SGD
    elif optimizer in ["sgd", "Sgd", "SGD"]:
        optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m in client_models] 
        if disc_method == "seperate" and lambda_disc > 0:
            optimizers_disc = [torch.optim.SGD(disc.parameters(), lr=lr) for disc in discriminators] 
    else:
        raise ValueError("Optimizer unknown.")
    
    #Each client updates its model locally on its own dataset
    print(kd_type)
    for r in range(rounds):
        t0 = time.time()
        
        for client_id in range(n_clients):
            #Setting up the local training
            model = client_models[client_id]
            model.train()
            opt = optimizers[client_id]
            if lambda_disc > 0:
                disc = discriminators[client_id]
                if disc_method == "seperate":
                    opt_disc = optimizers_disc[client_id]
            
            #Local update
            for e in range(epoch_per_round):
                for inputs, targets in train_dl_list[client_id]:
                    # Reset gradient
                    opt.zero_grad()
                    if disc_method == "seperate" and lambda_disc > 0:
                        opt_disc.zero_grad()
                    
                    # Local forward pass
                    features = model.features(inputs)
                    logits = model.classifier(features)
                        
                    # Optimization step
                    loss = criterion(logits, targets)
                    
                    # print(no_of_ones)
                    non_partner_client = None
                    if(len(non_partner_client_list[client_id])>0):
                        rand_idx = random.randrange(len(non_partner_client_list[client_id]))
                        non_partner_client = non_partner_client_list[client_id][rand_idx]
               
                    if lambda_kd > 0 and r%1 == 0:
                        # Compute estimated probabilities
                        rand_idx = random.randrange(n_clients-1)
                        if(rand_idx >= client_id):
                            rand_idx += 1
                        
                        if kd_type == "feature":
                            partner_teacher_data = feature_tracker.get_global_outputs(n_avg=n_avg, client_id="random",sel_cli=rand_idx).to(device)
                            _pull = torch.ones_like(targets.float())
                            _pull[(targets==0).nonzero().squeeze()] = 1
                            _pull[(targets==expertise[rand_idx]).nonzero().squeeze()] = 1
                            no_of_ones_pull = (_pull).sum(dim=0)
                            loss += lambda_kd  * torch.mean(torch.matmul( _pull[:,None].T,criterion_kd(features, partner_teacher_data[targets])))/no_of_ones_pull  
                        
                        elif kd_type == "output":
                            partner_teacher_data = logits_tracker.get_global_outputs(n_avg=n_avg, client_id="random",sel_cli=rand_idx).to(device)
                            _pull = torch.ones_like(targets.float())
                            _pull[(targets==0).squeeze()] = 1
                            _pull[(targets==expertise[rand_idx]).squeeze()] = 1
                            no_of_ones_pull = (_pull).sum(dim=0)
                            loss += lambda_kd  * torch.mean(torch.matmul( _pull[:,None].T,criterion_kd(logits, partner_teacher_data[targets])))/no_of_ones_pull  
                    if lambda_disc > 0 and False:
                        non_partner_teacher_data = feature_tracker.get_global_outputs(n_avg=n_avg, client_id="random",sel_cli=non_partner_client).to(device)
                        targets_global = torch.arange(meta["n_class"]).to(device)
                        scores, disc_targets = disc(features, non_partner_teacher_data, targets, targets_global)
                        _ = torch.tensor([[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
                        _[0][expertise[non_partner_client]] = 1.0
                        _[expertise[client_id]][expertise[non_partner_client]] = 1.0 
                        # print(_.shape,_[targets].shape)
                        _ = _[targets].flatten().to(device)
                        no_of_ones = (_).sum(dim=0)
                        loss += lambda_disc * torch.dot(criterion_disc(scores, disc_targets),_)/no_of_ones
                        
                    # Optimization step
                    loss.backward()
                    opt.step()
                    if disc_method == "seperate" and lambda_disc > 0:
                        opt_disc.step()
            torch.save(model.state_dict(),str(client_id)+".pt")
        # Use FedAvg on the classifer
        if fed_avg == "classifier" and r%5 == 0:
            # Aggregation (weighted average)
            global_parameters = global_model.classifier.state_dict()
            for k in global_parameters.keys():
                global_parameters[k] = torch.stack([(meta["n_local"][i] / meta["n_total"]) * client_models[i].classifier.state_dict()[k] 
                                                    for i in range(n_clients)], 0).sum(0)

            # Broadcast classifiers
            for model in client_models:
                model.classifier.load_state_dict(global_parameters)


        # Use FedAvg on the model (standard FL)
        elif fed_avg == "model" and r%1 == 0:
            # Aggregation (weighted average)
            global_parameters = global_model.state_dict()
            for k in global_parameters.keys():
                global_parameters[k] = torch.stack([(meta["n_local"][i] / meta["n_total"]) * client_models[i].state_dict()[k] 
                                                    for i in range(n_clients)], 0).sum(0)

            # Broadcast classifiers
            global_model.load_state_dict(global_parameters)
            for model in client_models:
                model.load_state_dict(global_parameters)

        #Tracking performance
        if (track_history and (r+1) % track_history == 0) or (r+1) == rounds:
            for client_id in range(n_clients):
                perf_trackers[client_id].new_eval(index=r+1)
                
        # Compute representations
        logits_tracker.new_round()
        feature_tracker.new_round()
        
        t1 = time.time()    
    print("\nFinal average performance:")
    print("Train acc",)
    print(f"Data size : {data_size} Collab Flag :{collab_flag}")
    for _ in range(meta["n_class"]):
        print(list([pt.perf_histories[f"Validation{_}"]["accuracy"][-1] for pt in perf_trackers])) 
        print("0",sum(list([pt.perf_histories[f"Validation{_}"]["accuracy"][-1] for pt in perf_trackers])[:int(n_clients/2)])*2/n_clients)
        print("1",sum(list([pt.perf_histories[f"Validation{_}"]["accuracy"][-1] for pt in perf_trackers])[int(n_clients/2):])*2/n_clients)
    # Saving plots if necessary
    hlp.plot_global_training_history(perf_trackers, metric="accuracy", title="Training history: Accuracy",
                                     savepath=os.path.join(fig_directory, "accuracy_history.png") if export_dir is not None else None)
    hlp.plot_global_training_history(perf_trackers, metric="loss", title="Training history: Loss",
                                     savepath=os.path.join(fig_directory, "loss_history.png") if export_dir is not None else None)
    plt.close("all")
    return perf_trackers, feature_tracker, logits_tracker

def benchmark(n_clients, dataset, model, alpha="uniform", rounds=100, 
              batch_size=8, epoch_per_round=1, lr=1e-4, optimizer="adam", feature_dim=200, 
              n_avg=None, fed_avg=None, lambda_kd=1.0, lambda_disc=1.0, sizes=None, reduced=False, 
              track_history=1, export_dir=None, data_dir="./data",
              device=None, seed=0):
    """Run an becnhmark (against fl, fd and il).
    
    Arguments:
        -n_clients: Number of clients
        -dataset: Classification dataset
        -model: Model architecture
        -alpha: Concentration parameter for data split
        -rounds: Number of communication rounds
        -batch_size: Mini-batch size
        -epoch_per_round: Number of epoch per communication round
        -lr: Learning rate
        -optimizer: Optimizer type
        -feature_dim: Number of feature
        -n_avg: Number of considered samples for the averaging
        -fed_avg: Use federated averaging for model/classifier/nothing
        -lambda_kd: Meta parameter for feature-based KD
        -lambda_disc: Meta parameter for contrastive lost
        -sizes: Dataset sizes
        -reduced: Reduction parameter for the train dataset
        -track_history: Intervall beween validation evaluation during training
        -export_dir: Folder to save plots/configuration/etc.
        -data_dir: Folder to load/download the data
        -device: Device to use for learning
        -seed: Seed for reproductability
    Return:
        - pt: List of performance tracker for base experiment
        - pt_fl: List of performance tracker for fl
        - pt_fd: List of performance tracker for fd
        - pt_il: List of performance tracker for il
    """
    
    print(40 * "*")
    print("Starting benchmark")
    print(40 * "*")
    pt, _ = run(n_clients=n_clients, dataset=dataset, model=model, alpha=alpha, rounds=rounds, 
                  batch_size=batch_size, epoch_per_round=epoch_per_round, lr=lr, optimizer=optimizer, 
                  feature_dim=feature_dim, n_avg=n_avg, lambda_kd=lambda_kd, lambda_disc=lambda_disc, 
                  sizes=sizes, fed_avg=fed_avg, reduced=reduced, track_history=track_history, 
                  export_dir=export_dir, data_dir=data_dir, device=device, seed=seed)
    print(20 * "*")
    pt_fd, _ = run(n_clients=n_clients, dataset=dataset, model=model, alpha=alpha, rounds=rounds, 
                batch_size=batch_size, epoch_per_round=epoch_per_round, lr=lr, optimizer=optimizer, 
                feature_dim=feature_dim, lambda_kd=lambda_kd, sizes=sizes, reduced=reduced, track_history=track_history,
                export_dir=export_dir, data_dir=data_dir, device=device, preset="fd", seed=seed)
    print(20 * "*")
    pt_il, _ = run(n_clients=n_clients, dataset=dataset, model=model, alpha=alpha, rounds=rounds, 
                batch_size=batch_size, epoch_per_round=epoch_per_round, lr=lr, optimizer=optimizer, 
                feature_dim=feature_dim, sizes=sizes, reduced=reduced, track_history=track_history, 
                export_dir=export_dir, data_dir=data_dir, device=device, preset="il", seed=seed)
    print(20 * "*")
    pt_fl, _ = run(n_clients=n_clients, dataset=dataset, model=model, alpha=alpha, rounds=rounds, 
                batch_size=batch_size, epoch_per_round=epoch_per_round, lr=lr, optimizer=optimizer, 
                feature_dim=feature_dim, sizes=sizes, reduced=reduced, track_history=track_history,
                export_dir=export_dir, data_dir=data_dir, device=device, preset="fl", seed=seed)


    
    print("Benchmark done.")
    return pt, pt_fl, pt_fd, pt_il

if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    args_dict = args.__dict__
    
    if args_dict["config_file"] is not None:
        f = open(args_dict["config_file"])
        config = json.load(f)
        f.close()
        
        
        for key, val in config.items():
            args_dict[key] = val
            
    del args_dict["config_file"]
    
    
    # Run experiments
    pt_list, ft, _ = run(**args_dict)
