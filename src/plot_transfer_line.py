import os, sys
import pickle
from collections import defaultdict
from itertools import permutations
import pandas as pd
import numpy as np
from lib.log import set_exp_logging
import matplotlib.pyplot as plt
import seaborn as sns
# plot setting
sns.set(style="white", font_scale=2.5)

plot_style = {
    "linestyle": "solid",
    "linewidth": 3,
    "marker": None,
    "markersize": 6,
}
rep_color = "#ffa500"
bre_color = "#0059ff"

def ax_setting(ax):
    ### spine setting
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('gray')
    ax.yaxis.grid(linestyle="solid", lw=1, alpha=1, color="lightgray")
    # ax.plot(np.linspace(8,370,1000), np.ones(1000)*100, 'w-', lw=2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", facecolor='white', edgecolor='none')

if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 6), facecolor="w")
    # datasets transferability
    t_rep_list, t_bre_list = [], [] 
    for method in ["care", "apricot", "arachne"]:
        dt_dir = f"/src/experiments/{method}/transferability"
        df_repair = pd.read_csv(os.path.join(dt_dir, "transfer_datasets-repair.csv"))
        df_break = pd.read_csv(os.path.join(dt_dir, "transfer_datasets-break.csv"))
        # 1列目はmethod名なので除く
        t_rep_list.append(df_repair.values[:, 1:].astype(np.float32))
        t_bre_list.append(df_break.values[:, 1:].astype(np.float32))
    t_rep = np.concatenate(t_rep_list, axis=1)
    t_bre = np.concatenate(t_bre_list, axis=1)
    # nanを除く
    t_rep = t_rep[~np.isnan(t_rep)]
    t_bre = t_bre[~np.isnan(t_bre)]
    # 1次元にして昇順ソート
    t_rep = np.sort(t_rep.flatten())
    t_bre = np.sort(t_bre.flatten())
    # t_rep, t_breそれぞれに1以上の値が含まれていたら, その値を1にする
    t_rep = np.where(t_rep > 1, 1, t_rep)
    t_bre = np.where(t_bre > 1, 1, t_bre)
    print(t_rep.shape)
    print(t_bre.shape)
    # plot
    ax = fig.add_subplot(1, 2, 1)
    # 横軸
    x = np.arange(len(t_rep)) / len(t_rep)
    # 折れ線描画
    # 多いのでマーカーを3つ飛ばしで描画する
    plt.plot(x, t_rep, label="repairs", color=rep_color, markevery=3, **plot_style)
    plt.plot(x, t_bre, label="breaks", color=bre_color, markevery=3, **plot_style)
    # 50%の縦線
    plt.vlines(0.5, min(min(t_rep), min(t_bre)), 1, linestyles="dashed", linewidth=3, color="red")
    # グラフのタイトルと軸ラベルを設定
    ax.set_title("(a) Transferability for Datasets")
    ax.set_xlabel("")
    ax.set_ylabel("Transferability Scores")
    ax.set_ylim(min(min(t_rep), min(t_bre)), None)
    ax.set_xlim(0, 1)
    ax_setting(ax)

    # methods transferability
    mt_dir = f"/src/experiments/method-transferability"
    df_repair = pd.read_csv(os.path.join(mt_dir, "all-repair.csv"))
    df_break = pd.read_csv(os.path.join(mt_dir, "all-break.csv"))
    # 1列目はmethod名なので除く
    t_rep = df_repair.values[:, 1:].astype(np.float32)
    t_bre = df_break.values[:, 1:].astype(np.float32)
    # nanを除く
    t_rep = t_rep[~np.isnan(t_rep)]
    t_bre = t_bre[~np.isnan(t_bre)]
    # 1次元にして昇順ソート
    t_rep = np.sort(t_rep.flatten())
    t_bre = np.sort(t_bre.flatten())
    # t_rep, t_breそれぞれに1以上の値が含まれていたら, その値を1にする
    t_rep = np.where(t_rep > 1, 1, t_rep)
    t_bre = np.where(t_bre > 1, 1, t_bre)
    print(t_rep.shape)
    print(t_bre.shape)
    # plot
    ax = fig.add_subplot(1, 2, 2)
    # 横軸
    x = np.arange(len(t_rep)) / len(t_rep)
    # 折れ線描画
    plt.plot(x, t_rep, label="repairs", color=rep_color, **plot_style)
    plt.plot(x, t_bre, label="breaks", color=bre_color, **plot_style)
    # 50%の縦線
    plt.vlines(0.5, min(min(t_rep), min(t_bre)), 1, linestyles="dashed", linewidth=3, color="red")
    # グラフのタイトルと軸ラベルを設定
    ax.set_title("(b) Transferability for Repair Methods")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(min(min(t_rep), min(t_bre)), None)
    ax.set_xlim(0, 1)
    ax_setting(ax)
    # グラフを表示
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.2)
    plt.savefig("./trans_results.pdf", bbox_inches="tight", dpi=300)
