import pandas as pd
import matplotlib.pyplot as plt


def contourplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
):
    # 横持ちにする
    df_z = df.set_index([y, x])[z].unstack()

    # matplotlibの入力の形にする
    x_unique = df_z.columns.values
    y_unique = df_z.index.values
    z_matrix = df_z.values

    fig, ax = plt.subplots()

    # 等高線を描画
    cs = ax.contour(
        x_unique,
        y_unique,
        z_matrix,
        colors="k",
    )
    # 線に数字を載せる
    ax.clabel(
        cs,
        fontsize=10,
    )
    # 色を付ける
    cs = ax.contourf(
        x_unique,
        y_unique,
        z_matrix,
        cmap="coolwarm",
    )
    # カラーバーを追加
    fig.colorbar(cs)
    # ラベルを追加
    ax.set(
        xlabel=x,
        ylabel=y,
        title=z,
    )
