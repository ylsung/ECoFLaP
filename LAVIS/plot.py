import matplotlib.pyplot as plt

import torch

import seaborn as sns
# sns.set_theme(style="darkgrid")

import yaml


def compare(item1):
    
    item1 = item1[0]
    
    if item1.startswith("visual_encoder"):
        return int(item1.split(".")[2])
    elif item1.startswith("t5_model"):
        score = int(item1.split(".")[3]) + 10000
        if "decoder" in item1:
            score += 5000
            
        return score
        

    return 1
    
    # if item1.startswith("vision_encoder") and item2.startswith("vision_encoder"):
    #     return int(item1.split(".")[2]) < int(item2.split(".")[2])
    # elif item1.startswith("t5_model") and item2.startswith("t5_model"):
    #     return int(item1.split(".")[3]) < int(item2.split(".")[3])
    # elif item1.startswith("t5_model"):
    #     return 0
    # elif item1.startswith("vision_encoder"):
    #     return 1
    
    # return 1

def sparsity_ratios():
    
    sparsity_dict = {
        "first-order grad": "sparsity_dict/cc3m-blipt5_wanda_pruner_0.5-1.0-1.0_aobd_sum0.6_block.yaml",
        "zeroth-order grad": "sparsity_dict/cc3m-blipt5_wanda_pruner_0.5-1.0-1.0_olmezo-gradient_sum0.6_block.yaml",
    }
    
    sparsities = []
    
    results = {
    }
    
    vit_t5_edge = -1
    for k, v  in sparsity_dict.items():
        with open(v, "r") as f:
            d = yaml.load(f, yaml.FullLoader)
            
            pair = []
            for i, (d_k, d_v) in enumerate(d.items()):
                pair.append([d_k, d_v])
                
            pair.sort(key=compare)
            
            names = [p[0] for p in pair]
            y = [p[1] for p in pair]
            
            for idx, n in enumerate(names):
                if vit_t5_edge == -1 and n.startswith("t5_model."):
                    vit_t5_edge = idx
                
            print(names)
            
            x = list(range(len(y)))

            results[k] = [x, y]
            
    print(vit_t5_edge)

    fig, ax = plt.subplots()
    
    colors = ["#B38100", "#597A46", "#B83154", "#5E7BA6", "#817BCD", "#66D6C0", "#F69863"]
    markers = ["o", "s", "^", "*", "D", "1", "p"]

    for i, (name, (x, y)) in enumerate(results.items()):
        plt.plot(x, y, label=name, color=colors[i], marker=markers[i], linewidth=3, markersize=0, linestyle="solid")


    y_min, y_max = ax.get_ylim()
    
    plt.vlines(vit_t5_edge, y_min, y_max, colors=["black"], linestyles="dashed")
    
    ax.text(vit_t5_edge + 10, y_min + 0.03, "Separation between language\nand vision model", color='black', fontsize=11)
    
    
    ax.set_facecolor("#DDE0E1")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.ylabel("Sparsity", fontsize=14)
    plt.xlabel("Layer index", fontsize=14)
    plt.legend(prop={'size': 14})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(alpha=0.5, linestyle="--")
    
    file_name = "tmp.pdf"

    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.)
    # 


def magnitude(file_dict):
    
    results = {
    }
    
    vit_t5_edge = -1
    for k, v  in file_dict.items():
        d = torch.load(v)
            
        pair = []
        for i, (d_k, d_v) in enumerate(d.items()):
            pair.append([d_k, d_v.float().abs().mean().item()]) # avg score
            
        pair.sort(key=compare)
        
        names = [p[0] for p in pair]
        y = [p[1] for p in pair]
        
        for idx, n in enumerate(names):
            if vit_t5_edge == -1 and n.startswith("t5_model."):
                vit_t5_edge = idx
                
        for n, _y in zip(names, y):
            print(n, _y)
                
        print(names)
        
        x = list(range(len(y)))

        results[k] = [x, y]
            
    print(vit_t5_edge)

    fig, ax = plt.subplots()
    
    colors = ["#B38100", "#597A46", "#B83154", "#5E7BA6", "#817BCD", "#66D6C0", "#F69863"]
    markers = ["o", "s", "^", "*", "D", "1", "p"]

    for i, (name, (x, y)) in enumerate(results.items()):
        plt.plot(x, y, label=name, color=colors[i], marker=markers[i], linewidth=3, markersize=0, linestyle="solid")


    y_min, y_max = ax.get_ylim()
    
    plt.vlines(vit_t5_edge, y_min, y_max, colors=["black"], linestyles="dashed")
    
    ax.text(vit_t5_edge + 10, y_max * 0.8, "Separation between language\nand vision model", color='black', fontsize=11)
    
    
    ax.set_facecolor("#DDE0E1")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.ylabel("Magnitude", fontsize=14)
    plt.xlabel("Layer index", fontsize=14)
    plt.legend(prop={'size': 14})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(alpha=0.5, linestyle="--")
    
    file_name = "tmp_mag.pdf"

    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.)    
    

def gradient(file_dict):
    
    results = {
    }
    
    vit_t5_edge = -1
    for k, v  in file_dict.items():
        d = torch.load(v)
            
        pair = []
        for i, (d_k, d_v) in enumerate(d.items()):
            pair.append([d_k, d_v.float().abs().mean().item()]) # avg score
            
        pair.sort(key=compare)
        
        names = [p[0] for p in pair]
        y = [p[1] for p in pair]
        
        for idx, n in enumerate(names):
            if vit_t5_edge == -1 and n.startswith("t5_model."):
                vit_t5_edge = idx
                
        for n, _y in zip(names, y):
            print(n, _y)
                
        print(names)
        
        x = list(range(len(y)))

        results[k] = [x, y]
            
    print(vit_t5_edge)

    fig, ax = plt.subplots()
    
    colors = ["#597A46", "#B38100", "#B83154", "#5E7BA6", "#817BCD", "#66D6C0", "#F69863"]
    markers = ["o", "s", "^", "*", "D", "1", "p"]

    for i, (name, (x, y)) in enumerate(results.items()):
        plt.plot(x, y, label=name, color=colors[i], marker=markers[i], linewidth=3, markersize=0, linestyle="solid")


    y_min, y_max = ax.get_ylim()
    
    plt.vlines(vit_t5_edge, y_min, y_max, colors=["black"], linestyles="dashed")
    
    ax.text(vit_t5_edge + 10, y_max * 0.8, "Separation between language\nand vision model", color='black', fontsize=11)
    
    
    ax.set_facecolor("#DDE0E1")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.ylabel("Gradient", fontsize=14)
    plt.xlabel("Layer index", fontsize=14)
    plt.legend(prop={'size': 14})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(alpha=0.5, linestyle="--")
    
    file_name = "tmp_grad.pdf"

    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.)    
    
    
def aobd():
    
    results = {
        "Magnitude x Gradient": []
    }
    
    mag = "pruned_checkpoint/magnitude.pth"
    grad = "pruned_checkpoint/first_order_grad.pth"

    mag = torch.load(mag)
    grad = torch.load(grad)
    
    aobd = {k: (mag[k].float().cpu().abs() * grad[k].float().cpu().abs()).mean().item() for k in mag}

    file_dict = {"Magnitude x Gradient": aobd}
    
    vit_t5_edge = -1
    for k, v  in file_dict.items():
            
        pair = []
        for i, (d_k, d_v) in enumerate(v.items()):
            pair.append([d_k, d_v])
            
        pair.sort(key=compare)
        
        names = [p[0] for p in pair]
        y = [p[1] for p in pair]
        
        for idx, n in enumerate(names):
            if vit_t5_edge == -1 and n.startswith("t5_model."):
                vit_t5_edge = idx
                
        for n, _y in zip(names, y):
            print(n, _y)
                
        print(names)
        
        x = list(range(len(y)))

        results[k] = [x, y]
            
    print(vit_t5_edge)

    fig, ax = plt.subplots()
    
    colors = ["#597A46", "#B38100", "#B83154", "#5E7BA6", "#817BCD", "#66D6C0", "#F69863"]
    markers = ["o", "s", "^", "*", "D", "1", "p"]

    for i, (name, (x, y)) in enumerate(results.items()):
        plt.plot(x, y, label=name, color=colors[i], marker=markers[i], linewidth=3, markersize=0, linestyle="solid")


    y_min, y_max = ax.get_ylim()
    
    plt.vlines(vit_t5_edge, y_min, y_max, colors=["black"], linestyles="dashed")
    
    ax.text(vit_t5_edge + 10, y_max * 0.8, "Separation between language\nand vision model", color='black', fontsize=11)
    
    
    ax.set_facecolor("#DDE0E1")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.ylabel("Magnitude x Gradient", fontsize=14)
    plt.xlabel("Layer index", fontsize=14)
    plt.legend(prop={'size': 14})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(alpha=0.5, linestyle="--")
    
    file_name = "tmp_aobd.pdf"

    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.)
    
    
def wanda():
    
    file_dict = {
        "Wanda": "importance_scores/cc3m-blipt5_wanda_pruner_0.5-1.0-1.0.pth",
    }
    
    results = {}
    
    vit_t5_edge = -1
    for k, v  in file_dict.items():
        d = torch.load(v)
            
        pair = []
        for i, (d_k, d_v) in enumerate(d.items()):
            pair.append([d_k, d_v]) # avg score
            
        pair.sort(key=compare)
        
        names = [p[0] for p in pair]
        y = [p[1] for p in pair]
        
        for idx, n in enumerate(names):
            if vit_t5_edge == -1 and n.startswith("t5_model."):
                vit_t5_edge = idx
                
        for n, _y in zip(names, y):
            print(n, _y)
                
        print(names)
        
        x = list(range(len(y)))

        results[k] = [x, y]
            
    print(vit_t5_edge)

    fig, ax = plt.subplots()
    
    colors = ["#597A46", "#B38100", "#B83154", "#5E7BA6", "#817BCD", "#66D6C0", "#F69863"]
    markers = ["o", "s", "^", "*", "D", "1", "p"]

    for i, (name, (x, y)) in enumerate(results.items()):
        plt.plot(x, y, label=name, color=colors[i], marker=markers[i], linewidth=3, markersize=0, linestyle="solid")


    y_min, y_max = ax.get_ylim()
    
    plt.vlines(vit_t5_edge, y_min, y_max, colors=["black"], linestyles="dashed")
    
    ax.text(vit_t5_edge + 10, y_max * 0.8, "Separation between language\nand vision model", color='black', fontsize=11)
    
    
    ax.set_facecolor("#DDE0E1")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.ylabel("Local Score of Wanda", fontsize=14)
    plt.xlabel("Layer index", fontsize=14)
    plt.legend(prop={'size': 14})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(alpha=0.5, linestyle="--")
    
    file_name = "tmp_wanda.pdf"

    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.)    
    
def sparsegpt():
    
    file_dict = {
        "SparseGPT": "importance_scores/cc3m-blipt5_sparsegpt_pruner_0.5-1.0-1.0.pth",
    }
    
    results = {}
    
    vit_t5_edge = -1
    for k, v  in file_dict.items():
        d = torch.load(v)
            
        pair = []
        for i, (d_k, d_v) in enumerate(d.items()):
            pair.append([d_k, d_v]) # avg score
            
        pair.sort(key=compare)
        
        names = [p[0] for p in pair]
        y = [p[1] for p in pair]
        
        for idx, n in enumerate(names):
            if vit_t5_edge == -1 and n.startswith("t5_model."):
                vit_t5_edge = idx
                
        for n, _y in zip(names, y):
            print(n, _y)
                
        print(names)
        
        x = list(range(len(y)))

        results[k] = [x, y]
            
    print(vit_t5_edge)

    fig, ax = plt.subplots()
    
    colors = ["#B83154", "#597A46", "#B38100", "#5E7BA6", "#817BCD", "#66D6C0", "#F69863"]
    markers = ["o", "s", "^", "*", "D", "1", "p"]

    for i, (name, (x, y)) in enumerate(results.items()):
        plt.plot(x, y, label=name, color=colors[i], marker=markers[i], linewidth=3, markersize=0, linestyle="solid")


    y_min, y_max = ax.get_ylim()
    
    plt.vlines(vit_t5_edge, y_min, y_max, colors=["black"], linestyles="dashed")
    
    ax.text(vit_t5_edge + 10, y_max * 0.8, "Separation between language\nand vision model", color='black', fontsize=11)
    
    
    ax.set_facecolor("#DDE0E1")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.ylabel("Local Score of SparseGPT", fontsize=14)
    plt.xlabel("Layer index", fontsize=14)
    plt.legend(prop={'size': 14})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    # plt.yscale("log")
    plt.grid(alpha=0.5, linestyle="--")
    
    file_name = "tmp_sparsegpt.pdf"

    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.)    


# magnitude({"Magnitude": "pruned_checkpoint/magnitude.pth"})
# aobd()
gradient({"Gradient": "pruned_checkpoint/first_order_grad.pth"})
# wanda()
# sparsegpt()
# sparsity_ratios()