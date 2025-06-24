<!-- [Previous content remains unchanged until the add_candlestick_trace function] -->

def add_candlestick_trace(fig, df, row):
    hover_texts = [
        "Date: {date}<br>Month: {month}<br>Open: ${open:.2f}<br>High: ${high:.2f}<br>Low: ${low:.2f}<br>Close: ${close:.2f}<br>Volume: {volume:,.0f}<br>RSI: {rsi:.2f}<br>RVOL: {rvol:.2f}".format(
            date=r['date'].strftime('%m-%d-%Y'), month=r['date'].strftime('%B'), open=r['open'], high=r['high'], low=r['low'], 
            close=r['close'], volume=r['volume'], rsi=r['rsi'], rvol=r['rvol']
        )
        for r in df.itertuples()
    ]
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Candlestick",
        increasing_line_color='#4CAF50', decreasing_line_color='#f44336',
        hovertext=hover_texts,
        hoverinfo='text',
        customdata=df.index
    ), row=row, col=1)
    if "Bollinger Bands" in show_indicators and 'ma20' in df.columns and 'std_dev' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['ma20'] + 2*df['std_dev'], name="Bollinger Upper", line=dict(color="#0288d1")), row=row, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['ma20'] - 2*df['std_dev'], name="Bollinger Lower", line=dict(color="#0288d1"), fill='tonexty', fillcolor='rgba(2,136,209,0.1)'), row=row, col=1)
    if "Ichimoku Cloud" in show_indicators:
        fig.add_trace(go.Scatter(x=df['date'], y=df['senkou_span_a'], name="Senkou Span A", line=dict(color="#4CAF50"), fill='tonexty', fillcolor='rgba(76,175,80,0.2)'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['senkou_span_b'], name="Senkou Span B", line=dict(color="#f44336"), fill='tonexty', fillcolor='rgba(244,67,54,0.2)'), row=row, col=1)
    if "Fibonacci" in show_indicators:
        for level, color in [('fib_236', '#ff9800'), ('fib_382', '#e91e63'), ('fib_50', '#9c27b0'), ('fib_618', '#3f51b5')]:
            fig.add_trace(go.Scatter(x=df['date'], y=df[level], name=f"Fib {level[-3:]}%", line=dict(color=color, dash='dash'),
                                     hovertext=[f"Fib {level[-3:]}%: ${x:.2f}" for x in df[level]], hoverinfo='text+x'), row=row, col=1)
    buy_signals = df[df['buy_signal'] == True]
    for _, signal_row in buy_signals.iterrows():
        fig.add_annotation(x=signal_row['date'], y=signal_row['high'], text="Buy", showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color="#000000"), row=row, col=1)
    if not buy_signals.empty:
        latest_buy = buy_signals.iloc[-1]
        risk = latest_buy['close'] - latest_buy['stop_loss']
        reward = latest_buy['take_profit'] - latest_buy['close']
        rr_ratio = reward / risk if risk > 0 else 'N/A'
        fig.add_hline(y=latest_buy['stop_loss'], line_dash="dash", line_color="#f44336", annotation_text="Stop-Loss", annotation_font_color="#000000", row=row, col=1)
        fig.add_hline(y=latest_buy['take_profit'], line_dash="dash", line_color="#4CAF50", annotation_text="Take-Profit", annotation_font_color="#000000", row=row, col=1)
        fig.add_trace(go.Scatter(x=[latest_buy['date'], latest_buy['date']], y=[latest_buy['stop_loss'], latest_buy['take_profit']],
                                 mode='lines', line=dict(color='rgba(76,175,80,0.2)'), fill='toself', fillcolor='rgba(76,175,80,0.2)',
                                 hovertext=[f"Risk-Reward Ratio: {rr_ratio:.2f}" if isinstance(rr_ratio, float) else f"Risk-Reward Ratio: {rr_ratio}"], hoverinfo='text', showlegend=False), row=row, col=1)

<!-- [Remaining content remains unchanged] -->
