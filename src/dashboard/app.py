import json, time
from collections import deque
from pathlib import Path
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px

STREAM_PATH = Path(__file__).resolve().parents[1] / 'data' / 'stream.jsonl'

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Real-Time Sentiment Dashboard"),
    html.Div(id='kpis', children=[
        html.Div(id='kpi_total'),
        html.Div(id='kpi_pos_rate'),
    ], style={'display':'flex','gap':'2rem'}),
    dcc.Graph(id='sentiment_over_time'),
    dcc.Graph(id='prob_hist'),
    dash_table.DataTable(id='recent_table', page_size=10),
    dcc.Interval(id='interval', interval=2000, n_intervals=0)
], style={'padding':'1rem','fontFamily':'sans-serif'})

def load_stream(max_rows=2000):
    if not STREAM_PATH.exists():
        return pd.DataFrame(columns=['ts','prob_pos','label','text','source'])
    records = []
    with open(STREAM_PATH, 'r', encoding='utf-8') as f:
        for line in f.readlines()[-max_rows:]:
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception:
                continue
    if not records:
        return pd.DataFrame(columns=['ts','prob_pos','label','text','source'])
    df = pd.DataFrame(records)
    df['time'] = pd.to_datetime(df['ts'], unit='s')
    return df

@app.callback(
    Output('kpi_total','children'),
    Output('kpi_pos_rate','children'),
    Output('sentiment_over_time','figure'),
    Output('prob_hist','figure'),
    Output('recent_table','data'),
    Output('recent_table','columns'),
    Input('interval','n_intervals')
)
def refresh(_):
    df = load_stream()
    total = len(df)
    pos_rate = df['label'].mean() if total else 0.0
    k1 = f"Events: {total}"
    k2 = f"Positive rate: {pos_rate*100:.1f}%"

    if total:
        by_min = df.set_index('time').resample('1min')['label'].mean().reset_index()
        fig_line = px.line(by_min, x='time', y='label', title='Positive rate by minute')
        fig_hist = px.histogram(df, x='prob_pos', nbins=20, title='Distribution of positive probability')
        recent = df.sort_values('time', ascending=False)[['time','label','prob_pos','text','source']].head(20)
    else:
        fig_line = px.line(pd.DataFrame({'time':[], 'label':[]}), x='time', y='label', title='Positive rate by minute')
        fig_hist = px.histogram(pd.DataFrame({'prob_pos':[]}), x='prob_pos', nbins=20, title='Distribution of positive probability')
        recent = pd.DataFrame(columns=['time','label','prob_pos','text','source'])

    columns=[{'name':c,'id':c} for c in recent.columns]
    return k1, k2, fig_line, fig_hist, recent.to_dict('records'), columns

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
