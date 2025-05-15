import polars as pl
import plotly.graph_objects as plt # type: ignore


df = pl.read_csv("test.tsv", has_header=True, separator="\t")


fig = plt.Figure()

fig.add_trace(
    plt.Scatter(
        x=df["loop_score"],
        y=df["xcorr"],
        mode="markers",
        marker=dict(
            color="blue",
        ),
        name="Precalulcated LoOP score with python",
    )
)

fig.add_trace(
    plt.Scatter(
        x=df["loop_score_rust"],
        y=df["xcorr"],
        mode="markers",
        marker=dict(
            color="red",
        ),
        name="LoOP of this library",
    )
)

fig.update_layout(
    title="LoOP score comparison",
    xaxis_title="xcorr",
    yaxis_title="LoOP score",
)

fig.show()