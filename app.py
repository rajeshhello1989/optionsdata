import matplotlib
# Set the Agg backend to prevent the "main thread is not in main loop" error
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template_string, request, url_for, redirect
import io
import base64
import datetime
import matplotlib.dates as mdates
import time
import pandas as pd
import numpy as np # Import numpy for data manipulation
from matplotlib.ticker import FuncFormatter # For Y-axis formatting
import liencompfyer as t1

# Initialize the Flask application
app = Flask(__name__)

# --- Dummy Data for Demonstration (Unchanged) ---
DUMMY_DATA_1 = {
    'candles': [[1760327100, 353, 385.4, 317.3, 342.75, 140025], [1760327400, 344.1, 352, 328.05, 339.6, 111075], [1760327700, 339.5, 341.95, 317.8, 320.35, 72225], [1760328000, 320, 323.45, 283, 289, 160950], [1760328300, 288.25, 324.8, 280, 319.5, 270900], [1760328600, 318.9, 332.95, 313, 330.7, 140250], [1760328900, 329.15, 349.35, 323.95, 342.85, 126900], [1760329200, 342.85, 343.75, 317.85, 321.65, 58050], [1760329500, 320, 330.85, 312.15, 326.4, 54825]],
    'code': 200, 's': 'ok'
}
DUMMY_DATA_2 = {
    'candles': [[1760327100, 7, 15.25, 7, 10.8, 11500650], [1760327400, 10.7, 12.15, 10, 10.4, 4426950], [1760327700, 10.45, 12, 10.35, 11.45, 3784275], [1760328000, 11.55, 14.85, 11.3, 13.35, 6396600], [1760328300, 13.6, 14.5, 9.75, 10.35, 5467425], [1760328600, 10.4, 11, 9.5, 10.2, 3616875], [1760328900, 10.35, 10.75, 8.65, 9.15, 3291150], [1760329200, 9.15, 10.7, 9.15, 10.25, 2425650], [1760329500, 10.3, 11, 9.7, 10.1, 2634525]],
    'code': 200, 's': 'ok'
}

# --- OPTIONS DATA (Unchanged) ---
OPTIONS_DATA = {'code': 200, 'data': {'callOi': 185190600, 'expiryData': [{'date': '14-10-2025', 'expiry': '1760436000'}, {'date': '20-10-2025', 'expiry': '1760954400'}, {'date': '28-10-2025', 'expiry': '1761645600'}, {'date': '04-11-2025', 'expiry': '1762250400'}, {'date': '11-11-2025', 'expiry': '1762855200'}, {'date': '25-11-2025', 'expiry': '1764064800'}, {'date': '30-12-2025', 'expiry': '1767088800'}, {'date': '31-03-2026', 'expiry': '1774951200'}, {'date': '30-06-2026', 'expiry': '1782813600'}, {'date': '29-09-2026', 'expiry': '1790676000'}, {'date': '29-12-2026', 'expiry': '1798538400'}, {'date': '29-06-2027', 'expiry': '1814263200'}, {'date': '28-12-2027', 'expiry': '1829988000'}, {'date': '27-06-2028', 'expiry': '1845712800'}, {'date': '26-12-2028', 'expiry': '1861437600'}, {'date': '26-06-2029', 'expiry': '1877162400'}, {'date': '24-12-2029', 'expiry': '1892800800'}, {'date': '25-06-2030', 'expiry': '1908612000'}], 'indiavixData': {'ask': 0, 'bid': 0, 'description': 'INDIAVIX-INDEX', 'ex_symbol': 'INDIAVIX', 'exchange': 'NSE', 'fyToken': '101000000026017', 'ltp': 11.16, 'ltpch': 0.15, 'ltpchp': 1.36, 'option_type': '', 'strike_price': -1, 'symbol': 'NSE:INDIAVIX-INDEX'}, 'optionsChain': [{'ask': 0, 'bid': 0, 'description': 'NIFTY50-INDEX', 'ex_symbol': 'NIFTY', 'exchange': 'NSE', 'fp': 25185.5, 'fpch': -123.8, 'fpchp': -0.49, 'fyToken': '101000000026000', 'ltp': 25145.5, 'ltpch': -81.85, 'ltpchp': -0.32, 'option_type': '', 'strike_price': -1, 'symbol': 'NSE:NIFTY50-INDEX'}, {'ask': 0.05, 'bid': 0, 'fyToken': '101125101442670', 'ltp': 0.05, 'ltpch': -5.7, 'ltpchp': -99.13, 'oi': 5555025, 'oich': -1942875, 'oichp': -25.91, 'option_type': 'PE', 'prev_oi': 7497900, 'strike_price': 24900, 'symbol': 'NSE:NIFTY25O1424900PE', 'volume': 536404575}, {'ask': 246, 'bid': 245.3, 'fyToken': '101125101442669', 'ltp': 246.1, 'ltpch': -105.05, 'ltpchp': -29.92, 'oi': 302850, 'oich': -54225, 'oichp': -15.19, 'option_type': 'CE', 'prev_oi': 357075, 'strike_price': 24900, 'symbol': 'NSE:NIFTY25O1424900CE', 'volume': 13907775}, {'ask': 0.05, 'bid': 0, 'fyToken': '101125101442672', 'ltp': 0.05, 'ltpch': -6.95, 'ltpchp': -99.29, 'oi': 3968625, 'oich': -921375, 'oichp': -18.84, 'option_type': 'PE', 'prev_oi': 4890000, 'strike_price': 24950, 'symbol': 'NSE:NIFTY25O1424950PE', 'volume': 708610500}, {'ask': 195.6, 'bid': 195.2, 'fyToken': '101125101442671', 'ltp': 195.35, 'ltpch': -107.45, 'ltpchp': -35.49, 'oi': 329325, 'oich': 220950, 'oichp': 203.88, 'option_type': 'CE', 'prev_oi': 108375, 'strike_price': 24950, 'symbol': 'NSE:NIFTY25O1424950CE', 'volume': 19457850}, {'ask': 0.05, 'bid': 0, 'fyToken': '101125101442676', 'ltp': 0.05, 'ltpch': -9.05, 'ltpchp': -99.45, 'oi': 8184300, 'oich': -5162300, 'oichp': -38.68, 'option_type': 'PE', 'prev_oi': 13346600, 'strike_price': 25000, 'symbol': 'NSE:NIFTY25O1425000PE', 'volume': 1289483250}, {'ask': 145.6, 'bid': 145.5, 'fyToken': '101125101442673', 'ltp': 145.6, 'ltpch': -107.2, 'ltpchp': -42.41, 'oi': 1034025, 'oich': -1081355, 'oichp': -51.12, 'option_type': 'CE', 'prev_oi': 2115380, 'strike_price': 25000, 'symbol': 'NSE:NIFTY25O1425000CE', 'volume': 154949700}, {'ask': 95.7, 'bid': 95.55, 'fyToken': '101125101442677', 'ltp': 95.55, 'ltpch': -109.95, 'ltpchp': -53.5, 'oi': 1216875, 'oich': 521625, 'oichp': 75.03, 'option_type': 'CE', 'prev_oi': 695250, 'strike_price': 25050, 'symbol': 'NSE:NIFTY25O1425050CE', 'volume': 276657975}, {'ask': 0.1, 'bid': 0.05, 'fyToken': '101125101442678', 'ltp': 0.1, 'ltpch': -12, 'ltpchp': -99.17, 'oi': 6897600, 'oich': 1472780, 'oichp': 27.15, 'option_type': 'PE', 'prev_oi': 5424820, 'strike_price': 25050, 'symbol': 'NSE:NIFTY25O1425050PE', 'volume': 1491789750}, {'ask': 0.05, 'bid': 0, 'fyToken': '101125101442684', 'ltp': 0.05, 'ltpch': -16.95, 'ltpchp': -99.71, 'oi': 12694875, 'oich': 1087775, 'oichp': 9.37, 'option_type': 'PE', 'prev_oi': 11607100, 'strike_price': 25100, 'symbol': 'NSE:NIFTY25O1425100PE', 'volume': 2291730000}, {'ask': 45.7, 'bid': 45.6, 'fyToken': '101125101442679', 'ltp': 45.75, 'ltpch': -115.25, 'ltpchp': -71.58, 'oi': 5883525, 'oich': 3700575, 'oichp': 169.52, 'option_type': 'CE', 'prev_oi': 2182950, 'strike_price': 25100, 'symbol': 'NSE:NIFTY25O1425100CE', 'volume': 1144206225}, {'ask': 4.5, 'bid': 4.4, 'fyToken': '101125101442686', 'ltp': 4.4, 'ltpch': -20.75, 'ltpchp': -82.5, 'oi': 14838525, 'oich': 4807925, 'oichp': 47.93, 'option_type': 'PE', 'prev_oi': 10030600, 'strike_price': 25150, 'symbol': 'NSE:NIFTY25O1425150PE', 'volume': 2832568425}, {'ask': 0.05, 'bid': 0, 'fyToken': '101125101442685', 'ltp': 0.05, 'ltpch': -118.85, 'ltpchp': -99.96, 'oi': 23567250, 'oich': 21533700, 'oichp': 1058.92, 'option_type': 'CE', 'prev_oi': 2033550, 'strike_price': 25150, 'symbol': 'NSE:NIFTY25O1425150CE', 'volume': 2413773150}, {'ask': 54.4, 'bid': 54.2, 'fyToken': '101125101442690', 'ltp': 54.2, 'ltpch': 15.7, 'ltpchp': 40.78, 'oi': 7350375, 'oich': -8633425, 'oichp': -54.01, 'option_type': 'PE', 'prev_oi': 15983800, 'strike_price': 25200, 'symbol': 'NSE:NIFTY25O1425200PE', 'volume': 1428934050}, {'ask': 0.05, 'bid': 0, 'fyToken': '101125101442687', 'ltp': 0.05, 'ltpch': -81.85, 'ltpchp': -99.94, 'oi': 22408725, 'oich': 11798625, 'oichp': 111.2, 'option_type': 'CE', 'prev_oi': 10610100, 'strike_price': 25200, 'symbol': 'NSE:NIFTY25O1425200CE', 'volume': 2498457300}, {'ask': 104.2, 'bid': 104.15, 'fyToken': '101125101442692', 'ltp': 104.05, 'ltpch': 44.35, 'ltpchp': 74.29, 'oi': 2177550, 'oich': -3225370, 'oichp': -59.7, 'option_type': 'PE', 'prev_oi': 5402920, 'strike_price': 25250, 'symbol': 'NSE:NIFTY25O1425250PE', 'volume': 569013150}, {'ask': 0.1, 'bid': 0.05, 'fyToken': '101125101442691', 'ltp': 0.05, 'ltpch': -53.9, 'ltpchp': -99.91, 'oi': 9716250, 'oich': 680100, 'oichp': 7.53, 'option_type': 'CE', 'prev_oi': 9036150, 'strike_price': 25250, 'symbol': 'NSE:NIFTY25O1425250CE', 'volume': 1353703275}, {'ask': 154.3, 'bid': 154.05, 'fyToken': '101125101442694', 'ltp': 154.05, 'ltpch': 64.05, 'ltpchp': 71.17, 'oi': 1792650, 'oich': -2282030, 'oichp': -56.01, 'option_type': 'PE', 'prev_oi': 4074680, 'strike_price': 25300, 'symbol': 'NSE:NIFTY25O1425300PE', 'volume': 419480775}, {'ask': 0.1, 'bid': 0.05, 'fyToken': '101125101442693', 'ltp': 0.05, 'ltpch': -33.7, 'ltpchp': -99.85, 'oi': 15067500, 'oich': -281400, 'oichp': -1.83, 'option_type': 'CE', 'prev_oi': 15348900, 'strike_price': 25300, 'symbol': 'NSE:NIFTY25O1425300CE', 'volume': 1197832350}, {'ask': 204.4, 'bid': 203.95, 'fyToken': '101125101442702', 'ltp': 204, 'ltpch': 77.6, 'ltpchp': 61.39, 'oi': 622875, 'oich': -1202775, 'oichp': -65.88, 'option_type': 'PE', 'prev_oi': 1825650, 'strike_price': 25350, 'symbol': 'NSE:NIFTY25O1425350PE', 'volume': 103026450}, {'ask': 0.05, 'bid': 0, 'fyToken': '101125101442695', 'ltp': 0.05, 'ltpch': -20.34, 'ltpchp': -99.75, 'oi': 6757725, 'oich': -3074925, 'oichp': -31.27, 'option_type': 'CE', 'prev_oi': 9832650, 'strike_price': 25350, 'symbol': 'NSE:NIFTY25O1425350CE', 'volume': 634849350}, {'ask': 254.45, 'bid': 254.3, 'fyToken': '101125101442704', 'ltp': 254.3, 'ltpch': 86.15, 'ltpchp': 51.23, 'oi': 383775, 'oich': -797625, 'oichp': -67.52, 'option_type': 'PE', 'prev_oi': 1181400, 'strike_price': 25400, 'symbol': 'NSE:NIFTY25O1425400PE', 'volume': 60465975}, {'ask': 0.1, 'bid': 0.05, 'fyToken': '101125101442703', 'ltp': 0.1, 'ltpch': -11.9, 'ltpchp': -99.17, 'oi': 7892775, 'oich': -6007925, 'oichp': -43.22, 'option_type': 'CE', 'prev_oi': 13900700, 'strike_price': 25400, 'symbol': 'NSE:NIFTY25O1425400CE', 'volume': 505736325}], 'putOi': 136227150}, 'message': '', 's': 'ok'}


# --- NEW ALERT CHECK FUNCTION (remains the same) ---
def check_for_cross_alert(data1, data2):
    """
    Checks if the last two data points indicate a recent crossing of the two lines.
    Returns: 'Cross Up', 'Cross Down', or None
    """
    candles1 = data1.get('candles', [])
    candles2 = data2.get('candles', [])

    if len(candles1) < 2 or len(candles2) < 2:
        return None

    A_prev = candles1[-2][4]
    A_curr = candles1[-1][4]

    B_prev = candles2[-2][4]
    B_curr = candles2[-1][4]

    A_is_above_B_curr = A_curr > B_curr
    A_is_above_B_prev = A_prev > B_prev

    if not A_is_above_B_prev and A_is_above_B_curr:
        return 'Cross Up'

    elif A_is_above_B_prev and not A_is_above_B_curr:
        return 'Cross Down'

    return None

# --- UPDATED FUNCTION FOR OI BAR CHART (Includes Annotations and LTP Line) ---
def create_oi_bar_chart(options_data, index_name, selected_expiry):
    """
    Generates a comparison bar chart of Open Interest (OI), Change in OI,
    and Previous OI for different strike prices for a selected expiry.
    """
    options_chain = options_data.get('data', {}).get('optionsChain', [])
    expiry_data = options_data.get('data', {}).get('expiryData', [])

    # Find the full expiry date string (e.g., '14-10-2025')
    expiry_date_str = next((item['date'] for item in expiry_data if item['expiry'] == selected_expiry), selected_expiry)

    # --- NIFTY LTP Extraction ---
    nifty_ltp = None
    index_entry = next((item for item in options_chain if item.get('symbol') == f'NSE:{index_name.upper()}-INDEX'), None)
    if index_entry:
        nifty_ltp = index_entry.get('ltp')

    # --- Filtering Logic (Find Symbol Fragment for Filtering) ---
    symbol_date_fragment = ''
    try:
        # Assuming the symbol format is 'NSE:NIFTY25O1424900PE', we extract '25O14'
        # Filter for entries that are actually options (have strike_price and option_type)
        first_option_symbol_entry = next(d for d in options_chain if d.get('strike_price', -1) != -1 and d.get('option_type'))
        first_option_symbol = first_option_symbol_entry['symbol']

        # Robustly determine the index name portion
        index_name_upper = index_name.upper()
        if index_name_upper in first_option_symbol:
            fragment_start = first_option_symbol.find(index_name_upper) + len(index_name_upper)
            # This logic assumes a fixed-length date fragment (e.g., 5 chars like '25O14')
            symbol_date_fragment = first_option_symbol[fragment_start:fragment_start+5]

    except (StopIteration, IndexError, TypeError):
        # Fallback if the data structure is inconsistent or empty
        pass

    # Filter based on the symbol fragment
    oi_data = [d for d in options_chain
               if d.get('strike_price', -1) != -1
               and d.get('option_type') in ['CE', 'PE']
               and symbol_date_fragment in d.get('symbol', '')]

    if not oi_data:
        return None

    # Group by strike price and option type
    df = pd.DataFrame(oi_data)

    # Filter for the main option types (CE and PE) and relevant columns
    df_filtered = df[df['option_type'].isin(['CE', 'PE'])].rename(
        columns={'strike_price': 'Strike', 'oi': 'OI', 'oich': 'OI_Ch', 'prev_oi': 'Prev_OI', 'option_type': 'Type'}
    )

    # Pivot the table to have Strikes as index and CE/PE as columns
    oi_pivot = df_filtered.pivot(index='Strike', columns='Type', values=['OI', 'OI_Ch', 'Prev_OI']).fillna(0)

    # Flatten the columns for easier access: (OI, CE) -> OI_CE
    oi_pivot.columns = ['_'.join(col).strip() for col in oi_pivot.columns.values]

    strikes = oi_pivot.index.tolist()

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))

    # Determine a good bar width based on strike interval
    strike_interval = strikes[1] - strikes[0] if len(strikes) > 1 else 50
    bar_width = strike_interval * 0.35
    bar_offset_ch = bar_width * 0.6 # offset for OI Change bars

    # --- Plot 1: Current OI (Green/Red) ---
    ax.bar(strikes, oi_pivot['OI_CE'], width=-bar_width, align='edge', color='#FF5255', alpha=0.8, label='Call OI (Current)')
    ax.bar(strikes, oi_pivot['OI_PE'], width=bar_width, align='edge', color='#00C853', alpha=0.8, label='Put OI (Current)')

    # --- Plot 2: Change in OI (Blue for positive, Dark for negative) ---
    # CE OI Change (plots on the Call side)
    ce_ch_color = np.where(oi_pivot['OI_Ch_CE'] > 0, '#00B0FF', '#c91caf')
    # Use absolute value for height but map color based on sign
    ax.bar(np.array(strikes) - bar_width + bar_offset_ch, np.abs(oi_pivot['OI_Ch_CE']), width=bar_width*0.5, color=ce_ch_color, label='Call OI Change')

    # PE OI Change (plots on the Put side)
    pe_ch_color = np.where(oi_pivot['OI_Ch_PE'] > 0, '#00B0FF', '#b6c41a')
    ax.bar(np.array(strikes) + bar_offset_ch, np.abs(oi_pivot['OI_Ch_PE']), width=bar_width*0.5, color=pe_ch_color, label='Put OI Change')

    # --- Plot 3: Previous OI (Markers) ---
    ax.scatter(np.array(strikes) - bar_width, oi_pivot['Prev_OI_CE'], color='#FFD700', marker='D', s=40, label='Previous Call OI', zorder=3)
    ax.scatter(np.array(strikes) + bar_width, oi_pivot['Prev_OI_PE'], color='#FFA500', marker='X', s=40, label='Previous Put OI', zorder=3)

    # --- CRITICAL: LTP Vertical Line (The "Live" Reference Point) ---
    if nifty_ltp is not None:
        ax.axvline(x=nifty_ltp, color='#90CAF9', linestyle='--', linewidth=2, label=f'{index_name} LTP ({nifty_ltp:,.2f})', zorder=1)

    # --- CRITICAL: Max OI Annotations (The "Tooltip-Like" Highlights) ---
    if not oi_pivot.empty:
        # Find the strikes with the highest current OI
        max_call_oi_strike = oi_pivot['OI_CE'].idxmax()
        max_put_oi_strike = oi_pivot['OI_PE'].idxmax()

        # Annotate Max Call OI (Resistance)
        max_ce_y = oi_pivot.loc[max_call_oi_strike]['OI_CE']
        if max_ce_y > 0:
            ax.annotate(f'Max Call OI\n(Resistance: {max_call_oi_strike})',
                        xy=(max_call_oi_strike - bar_width, max_ce_y),
                        xytext=(-20, 20), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, color='white',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#FF5255'),
                        bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.8, ec="#FF5255"))

        # Annotate Max Put OI (Support)
        max_pe_y = oi_pivot.loc[max_put_oi_strike]['OI_PE']
        if max_pe_y > 0:
            ax.annotate(f'Max Put OI\n(Support: {max_put_oi_strike})',
                        xy=(max_put_oi_strike + bar_width, max_pe_y),
                        xytext=(20, 20), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, color='white',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color='#00C853'),
                        bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.8, ec="#00C853"))


    # Formatting
    ax.set_title(f'Open Interest for {index_name} (Expiry: {expiry_date_str})', fontsize=18, pad=15, color='white')
    ax.set_xlabel('Strike Price', fontsize=14, color='white')
    ax.set_ylabel('Open Interest (Contracts)', fontsize=14, color='white')

    ax.set_xticks(strikes)
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')

    # Custom y-axis formatter (to show millions)
    def million_formatter(x, pos):
        return f'{x/1000000:.1f}M'
    ax.yaxis.set_major_formatter(FuncFormatter(million_formatter))

    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', ncol=3, facecolor='black', framealpha=0.8, fontsize=10)

    # Add a note for OI Change colors
    fig.text(0.5, 0.01, 'Note: Blue = Net New OI (+ve Change); Darker Shades = OI Unwinding (-ve Change)',
             ha='center', fontsize=10, color='#90CAF9')

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    # --- Save Plot to Memory and Encode ---
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', transparent=False, dpi=120)
    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return plot_data

# --- Rest of the file remains the same ---
# ... (The rest of the code for create_comparison_chart, get_chart_dpi, get_refresh_interval_seconds, and the Flask routes) ...


def create_comparison_chart(data_1, data_2, symbol1='Symbol 1', symbol2='Symbol 2', resolution='5 mins'):
    candles_1 = data_1.get('candles', [])
    timestamps_1 = [datetime.datetime.fromtimestamp(c[0]) for c in candles_1]
    closing_prices_1 = [c[4] for c in candles_1]

    candles_2 = data_2.get('candles', [])
    timestamps_2 = [datetime.datetime.fromtimestamp(c[0]) for c in candles_2]
    closing_prices_2 = [c[4] for c in candles_2]

    if not timestamps_1 and not timestamps_2:
        return None

    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(14, 7))

    has_ax1 = bool(timestamps_1)
    if has_ax1:
        color_1 = '#00C853'
        ax1.set_ylabel(f'Price ({symbol1})', color=color_1, fontsize=14)
        ax1.plot(timestamps_1, closing_prices_1, color=color_1, linewidth=2, label=f'{symbol1} Price')
        ax1.tick_params(axis='y', labelcolor=color_1)
        ax1.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)

    has_ax2_data = bool(timestamps_2)
    has_twin_ax = False

    if has_ax2_data:
        if has_ax1:
            ax2 = ax1.twinx()
            color_2 = '#FF5255'
            ax2.set_ylabel(f'Price ({symbol2})', color=color_2, fontsize=14)
            ax2.plot(timestamps_2, closing_prices_2, color=color_2, linewidth=2, linestyle='--', label=f'{symbol2} Price')
            ax2.tick_params(axis='y', labelcolor=color_2)
            has_twin_ax = True
        else:
            color_2 = '#FF5255'
            ax1.set_ylabel(f'Price ({symbol2})', color=color_2, fontsize=14)
            ax1.plot(timestamps_2, closing_prices_2, color=color_2, linewidth=2, linestyle='-', label=f'{symbol2} Price')
            ax1.tick_params(axis='y', labelcolor=color_2)
            ax1.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)

    ax1.set_xlabel(f'Time ({resolution} Intervals)', fontsize=14, color='white')

    try:
        res_minutes = int(resolution.split()[0])
    except:
        res_minutes = 5

    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=res_minutes))
    ax1.tick_params(axis='x', rotation=45, colors='white')

    lines, labels = [], []
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines.extend(lines1)
    labels.extend(labels1)

    if has_twin_ax:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines.extend(lines2)
        labels.extend(labels2)

    if lines:
        ax1.legend(lines, labels, loc='upper left', facecolor='black', framealpha=0.8, fontsize=12)

    fig.autofmt_xdate(rotation=45)
    ax1.set_title(f'Price Comparison: {symbol1} vs {symbol2} ({resolution})', fontsize=18, pad=15, color='white')
    fig.tight_layout()

    img_buffer = io.BytesIO()
    dynamic_dpi = get_chart_dpi(resolution)
    plt.savefig(img_buffer, format='png', transparent=False, dpi=dynamic_dpi)

    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return plot_data

def get_chart_dpi(resolution):
    """Sets a DPI for Matplotlib based on the selected resolution for quality control."""
    resolution = resolution.lower()
    if 'day' in resolution:
        return 150
    elif 'hour' in resolution:
        return 120
    elif 'min' in resolution:
        return 100
    return 100

def get_refresh_interval_seconds(resolution_str):
    """Converts a resolution string (e.g., '5 mins') into a refresh interval in seconds."""
    try:
        parts = resolution_str.split()
        if len(parts) == 2:
            num = int(parts[0])
            unit = parts[1].lower()

            if 'min' in unit:
                interval_s = num * 60
            elif 'hour' in unit:
                interval_s = num * 3600
            elif 'day' in unit:
                interval_s = num * 86400
            else:
                interval_s = 5

            return max(interval_s, 5)

    except:
        pass
    return 5
# --- END Utility Functions ---


@app.route('/', methods=['GET', 'POST'])
def home():
    plot_data = None
    oi_plot_data = None
    cross_alert = None

    # Get metadata from the global OPTIONS_DATA for default values and dropdown
    expiry_data = OPTIONS_DATA.get('data', {}).get('expiryData', [])
    default_expiry = expiry_data[0]['expiry'] if expiry_data else 'N/A'

    # Attempt to derive default index name (e.g., NIFTY50)
    options_chain = OPTIONS_DATA.get('data', {}).get('optionsChain', [])
    default_index = 'NIFTY50'
    if options_chain and 'description' in options_chain[0]:
        default_index = options_chain[0]['description'].split('-')[0]

    # Initialize input values
    input_values = {
        'token': 'YOUR_ACCESS_TOKEN',
        'redirect_uri': 'http://localhost:8000/auth',
        'client_id': 'YOUR_CLIENT_ID',
        'secret_key': 'YOUR_SECRET_KEY',
        'symbol1': 'ACC',
        'symbol2': 'RELIANCE',
        'days': '1',
        'resolution': '5 mins',
        'refresh_interval': '300',
        'index_name': default_index,
        'selected_expiry': default_expiry
    }

    # Initialize data snapshot values
    latest_price_1 = 'N/A'
    latest_price_2 = 'N/A'
    total_call_oi = OPTIONS_DATA.get('data', {}).get('callOi', 'N/A')
    total_put_oi = OPTIONS_DATA.get('data', {}).get('putOi', 'N/A')

    # Initialize Max OI Snapshot values
    max_call_oi_strike = 'N/A'
    max_call_oi_value = 'N/A'
    max_put_oi_strike = 'N/A'
    max_put_oi_value = 'N/A'

    data1 = DUMMY_DATA_1
    data2 = DUMMY_DATA_2
    options_data = OPTIONS_DATA


    # Handle form submission
    if request.method == 'POST':
        # Read all form fields
        input_values['token'] = request.form.get('token', '')
        input_values['redirect_uri'] = request.form.get('redirect_uri', '')
        input_values['client_id'] = request.form.get('client_id', '')
        input_values['secret_key'] = request.form.get('secret_key', '')
        input_values['symbol1'] = request.form.get('symbol1', 'SYM1')
        input_values['symbol2'] = request.form.get('symbol2', 'SYM2')
        input_values['days'] = request.form.get('days', '1')
        input_values['resolution'] = request.form.get('resolution', '5 mins')
        input_values['index_name'] = request.form.get('index_name', default_index)
        input_values['selected_expiry'] = request.form.get('selected_expiry', default_expiry)

        # Handle refresh interval
        manual_refresh = request.form.get('refresh_interval', '')
        default_interval_s = get_refresh_interval_seconds(input_values['resolution'])

        try:
            input_values['refresh_interval'] = str(max(int(manual_refresh), 5))
        except ValueError:
            input_values['refresh_interval'] = str(default_interval_s)

        # --- Data Processing ---

        res_data=t1.responseData(input_values['redirect_uri'],input_values['client_id'],input_values['secret_key'],
                              input_values['token'],input_values['symbol1'] ,input_values['symbol2'],input_values['days'],
                                 input_values['resolution'], input_values['index_name'] )
        print(res_data)
        if(res_data[0] is None or res_data[1] is None or res_data[2] is None):
            data1 = DUMMY_DATA_1
            data2 = DUMMY_DATA_2
            options_data = OPTIONS_DATA
        else:
            data1 = res_data[0]
            data2 = res_data[1]
            options_data = res_data[2]
        total_call_oi = options_data.get('data', {}).get('callOi', 'N/A')
        total_put_oi = options_data.get('data', {}).get('putOi', 'N/A')

        # Latest Price Snapshot
        latest_price_1 = data1['candles'][-1][4] if data1.get('candles') and data1['candles'] else 'N/A'
        latest_price_2 = data2['candles'][-1][4] if data2.get('candles') and data2['candles'] else 'N/A'
        print(latest_price_1,latest_price_2)

        # Check for the crossing alert
        cross_alert = check_for_cross_alert(data1, data2)

        # --- NEW: Extracting Max OI Strikes and Values for Snapshot ---
        options_chain_snapshot = options_data.get('data', {}).get('optionsChain', [])
        call_oi_entries = [d for d in options_chain_snapshot if d.get('option_type') == 'CE' and d.get('oi') is not None and d.get('strike_price') is not None]
        put_oi_entries = [d for d in options_chain_snapshot if d.get('option_type') == 'PE' and d.get('oi') is not None and d.get('strike_price') is not None]

        if call_oi_entries:
            max_ce_entry = max(call_oi_entries, key=lambda x: x['oi'])
            max_call_oi_strike = max_ce_entry.get('strike_price', 'N/A')
            max_call_oi_value = max_ce_entry.get('oi', 'N/A')

        if put_oi_entries:
            max_pe_entry = max(put_oi_entries, key=lambda x: x['oi'])
            max_put_oi_strike = max_pe_entry.get('strike_price', 'N/A')
            max_put_oi_value = max_pe_entry.get('oi', 'N/A')
        # --- END NEW MAX OI EXTRACTION ---


        # Generate Price Comparison Chart
        plot_data = create_comparison_chart(
            data1,
            data2,
            input_values['symbol1'],
            input_values['symbol2'],
            input_values['resolution']
        )

        # Generate OI Bar Chart
        oi_plot_data = create_oi_bar_chart(
            options_data,
            input_values['index_name'],
            input_values['selected_expiry']
        )

    else:
        # Initial GET request
        default_interval_s = get_refresh_interval_seconds(input_values['resolution'])
        input_values['refresh_interval'] = str(default_interval_s)

        # Latest Price Snapshot for initial load
        latest_price_1 = data1['candles'][-1][4] if data1.get('candles') and data1['candles'] else 'N/A'
        latest_price_2 = data2['candles'][-1][4] if data2.get('candles') and data2['candles'] else 'N/A'

        # --- NEW: Extracting Max OI Strikes and Values for Snapshot (Initial Load) ---
        options_chain_snapshot = options_data.get('data', {}).get('optionsChain', [])
        call_oi_entries = [d for d in options_chain_snapshot if d.get('option_type') == 'CE' and d.get('oi') is not None and d.get('strike_price') is not None]
        put_oi_entries = [d for d in options_chain_snapshot if d.get('option_type') == 'PE' and d.get('oi') is not None and d.get('strike_price') is not None]

        if call_oi_entries:
            max_ce_entry = max(call_oi_entries, key=lambda x: x['oi'])
            max_call_oi_strike = max_ce_entry.get('strike_price', 'N/A')
            max_call_oi_value = max_ce_entry.get('oi', 'N/A')

        if put_oi_entries:
            max_pe_entry = max(put_oi_entries, key=lambda x: x['oi'])
            max_put_oi_strike = max_pe_entry.get('strike_price', 'N/A')
            max_put_oi_value = max_pe_entry.get('oi', 'N/A')
        # --- END NEW MAX OI EXTRACTION ---

        # Generate initial plots
        plot_data = create_comparison_chart(data1, data2)
        oi_plot_data = create_oi_bar_chart(options_data, input_values['index_name'], input_values['selected_expiry'])


    refresh_interval_ms = int(input_values['refresh_interval']) * 1000

    # --- HTML Template (UPDATED to include Max OI in Snapshot) ---
    template_html = '''
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Price Comparison Dashboard</title>
            <style>
              body {
                background-color: #121212;
                color: #e0e0e0;
                font-family: sans-serif;
                margin: 0;
                padding: 20px;
              }
              .header {
                text-align: center;
                margin-bottom: 20px;
                color: #ffffff;
              }
              .dashboard-container {
                display: flex;
                flex-direction: column;
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
              }
              .controls {
                background-color: #1e1e1e;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
              }
              .form-row {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                align-items: flex-end;
                width: 100%;
                margin-top: 15px;
              }
              .input-group {
                display: flex;
                flex-direction: column;
                min-width: 150px;
                flex-grow: 1;
              }
              .input-group.lg {
                flex-grow: 2;
                min-width: 250px;
              }
              label {
                margin-bottom: 5px;
                color: #90CAF9;
                font-weight: bold;
              }
              input[type="text"], input[type="number"], select {
                padding: 10px;
                border: 1px solid #333;
                border-radius: 5px;
                background-color: #2c2c2c;
                color: #e0e0e0;
                min-height: 40px;
              }
              button {
                padding: 10px 20px;
                background-color: #00C853;
                color: black;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
                transition: background-color 0.3s;
                min-height: 40px; 
                flex-grow: 0.5;
              }
              button:hover {
                background-color: #009624;
              }
              .chart-container {
                background-color: #1e1e1e;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                text-align: center;
              }
              img {
                  max-width: 100%;
                  height: auto;
                  display: block;
                  margin: 0 auto;
              }
              /* --- ALERT & SNAPSHOT STYLES --- */
              .alert-box {
                  padding: 15px;
                  margin-bottom: 20px;
                  border-radius: 8px;
                  font-weight: bold;
                  text-align: center;
                  animation: pulse 1s infinite alternate;
              }
              .alert-up {
                  background-color: #00C853; /* Green */
                  color: black;
              }
              .alert-down {
                  background-color: #FF5255; /* Red */
                  color: black;
              }
              .data-snapshot {
                  background-color: #1e1e1e;
                  padding: 15px 20px;
                  border-radius: 15px;
                  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
                  text-align: left;
              }
              .snapshot-row {
                  display: flex;
                  gap: 40px;
                  margin-top: 10px;
                  justify-content: space-around;
              }
            </style>
          </head>
          <body>
            <div class="header">
              <h1>Dual Price Series Chart Dashboard 📈</h1>
              <p>Parameters for API Configuration and Data Request</p>
            </div>
            
            <div class="dashboard-container">
                <div class="controls">
                    <form method="POST" id="dataForm">
                        <h2>API Configuration</h2>
                        <div class="form-row">
                            <div class="input-group lg">
                                <label for="client_id">Client ID:</label>
                                <input type="text" id="client_id" name="client_id" value="{{ input_values['client_id'] }}" required>
                            </div>
                            <div class="input-group lg">
                                <label for="secret_key">Secret Key:</label>
                                <input type="text" id="secret_key" name="secret_key" value="{{ input_values['secret_key'] }}" required>
                            </div>
                            <div class="input-group lg">
                                <label for="redirect_uri">Redirect URI:</label>
                                <input type="text" id="redirect_uri" name="redirect_uri" value="{{ input_values['redirect_uri'] }}" required>
                            </div>
                        </div>

                        <h2>Price Comparison Data Request</h2>
                        <div class="form-row">
                            <div class="input-group lg">
                                <label for="token">Access Token:</label>
                                <input type="text" id="token" name="token" value="{{ input_values['token'] }}" required>
                            </div>
                            <div class="input-group">
                                <label for="symbol1">Symbol 1:</label>
                                <input type="text" id="symbol1" name="symbol1" value="{{ input_values['symbol1'] }}" required>
                            </div>
                            <div class="input-group">
                                <label for="symbol2">Symbol 2:</label>
                                <input type="text" id="symbol2" name="symbol2" value="{{ input_values['symbol2'] }}" required>
                            </div>
                        </div>

                        <div class="form-row">
                             <div class="input-group">
                                <label for="days">Data Days:</label>
                                <input type="number" id="days" name="days" value="{{ input_values['days'] }}" min="1" required>
                            </div>
                            <div class="input-group">
                                <label for="resolution">Resolution (Chart Bars):</label>
                                                        <select id="resolution" name="resolution">
                           <option value="1" {% if input_values['resolution'] == '1' %}selected{% endif %}>1 min</option>
                           <option value="5" {% if input_values['resolution'] == '5' %}selected{% endif %}>5 mins</option>
                           <option value="15" {% if input_values['resolution'] == '15' %}selected{% endif %}>15 mins</option>
                           <option value="30" {% if input_values['resolution'] == '30' %}selected{% endif %}>30 mins</option>
                           <option value="60" {% if input_values['resolution'] == '60' %}selected{% endif %}>1 hour</option>
                           <option value="1D" {% if input_values['resolution'] == '1D' %}selected{% endif %}>1 day</option>
                        </select>
                            </div>
                             <div class="input-group">
                                <label for="refresh_interval">Refresh Interval (Secs):</label>
                                <input type="number" id="refresh_interval" name="refresh_interval" value="{{ input_values['refresh_interval'] }}" min="5" required>
                            </div>
                        </div>

                        <h2>Options Data Request</h2>
                        <div class="form-row">
                             <div class="input-group">
                                <label for="index_name">Index Name:</label>
                                <input type="text" id="index_name" name="index_name" value="{{ input_values['index_name'] }}" required>
                            </div>
                             <div class="input-group">
                                <label for="selected_expiry">Expiry Date:</label>
                                <select id="selected_expiry" name="selected_expiry">
                                    {% for exp in expiry_data %}
                                        <option value="{{ exp['expiry'] }}" 
                                                {% if input_values['selected_expiry'] == exp['expiry'] %}selected{% endif %}>
                                            {{ exp['date'] }}
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>
                            <button type="submit">Submit & Plot</button>
                        </div>

                    </form>
                </div>
                
                {% if plot_data %}
                
                <div class="data-snapshot">
                    <h3 style="color: #00B0FF; margin-top: 0;">📊 Latest Data Snapshot</h3>
                    <div class="snapshot-row">
                        <p><strong>{{ input_values['symbol1'] }} (Last Close):</strong> <span style="font-size: 1.2em; color: #00C853;">{{ latest_price_1 }}</span></p>
                        <p><strong>{{ input_values['symbol2'] }} (Last Close):</strong> <span style="font-size: 1.2em; color: #FF5255;">{{ latest_price_2 }}</span></p>
                    </div>
                    <div class="snapshot-row">
                        <p><strong>Total Call OI:</strong> <span style="font-size: 1.2em; color: #FF5255;">{{ "{:,}".format(total_call_oi) if total_call_oi != 'N/A' and total_call_oi is not none else 'N/A' }}</span></p>
                        <p><strong>Total Put OI:</strong> <span style="font-size: 1.2em; color: #00C853;">{{ "{:,}".format(total_put_oi) if total_put_oi != 'N/A' and total_put_oi is not none else 'N/A' }}</span></p>
                    </div>
                    
                    <hr style="border-top: 1px solid #333; margin: 10px 0;">
                    <h4 style="color: #FFD700; margin-bottom: 5px;">Max Open Interest (Resistance & Support)</h4>
                    <div class="snapshot-row">
                        <p>
                            <strong>Max Call OI (Resistance):</strong> 
                            <span style="font-size: 1.1em; color: #FF5255;">
                                {{ "{:,}".format(max_call_oi_value) if max_call_oi_value != 'N/A' and max_call_oi_value is not none else 'N/A' }} 
                                {% if max_call_oi_strike != 'N/A' %} @ Strike: {{ max_call_oi_strike }}{% endif %}
                            </span>
                        </p>
                        <p>
                            <strong>Max Put OI (Support):</strong> 
                            <span style="font-size: 1.1em; color: #00C853;">
                                {{ "{:,}".format(max_put_oi_value) if max_put_oi_value != 'N/A' and max_put_oi_value is not none else 'N/A' }} 
                                {% if max_put_oi_strike != 'N/A' %} @ Strike: {{ max_put_oi_strike }}{% endif %}
                            </span>
                        </p>
                    </div>
                    </div>

                {% if cross_alert == 'Cross Up' %}
                <div class="alert-box alert-up">
                    ⬆️ CROSS UP ALERT! {{ input_values['symbol1'] }} has crossed **ABOVE** {{ input_values['symbol2'] }}! 🚀
                </div>
                {% elif cross_alert == 'Cross Down' %}
                <div class="alert-box alert-down">
                    ⬇️ CROSS DOWN ALERT! {{ input_values['symbol1'] }} has crossed **BELOW** {{ input_values['symbol2'] }}! 📉
                </div>
                {% endif %}
                
                <div class="chart-container">
                    <h2>Price Comparison Chart</h2>
                    <p>Last Updated: {{ now }}. Refreshing every {{ input_values['refresh_interval'] }} seconds. Chart DPI: {{ get_chart_dpi(input_values['resolution']) }}.</p>
                    <img src="data:image/png;base64,{{ plot_data }}" alt="Price Comparison Chart">
                </div>
                
                {% if oi_plot_data %}
                <div class="chart-container">
                    <h2>Options Open Interest (OI) Bar Chart (with Support/Resistance Annotations)</h2>
                    <img src="data:image/png;base64,{{ oi_plot_data }}" alt="Open Interest Bar Chart">
                </div>
                {% endif %}
                
                {% elif request.method == 'POST' %}
                <div class="chart-container" style="color: #FF5255;">
                    <p>Error: Could not generate chart. Please check the inputs or the data source (Dummy data used here).</p>
                </div>
                {% else %}
                <div class="chart-container">
                    <p>Enter parameters above and click 'Submit & Plot' to view the comparison chart.</p>
                </div>
                {% endif %}
            </div>

            <script>
                // Correct for Python's resolution string inconsistency in the template
                const resolutionSelect = document.getElementById('resolution');
                let selectedResolutionValue = resolutionSelect.value;
                if (!selectedResolutionValue.includes(' ')) {
                    if (selectedResolutionValue === '1D') {
                        resolutionSelect.value = '1 day';
                    } else if (selectedResolutionValue === '1' || selectedResolutionValue === '5' || selectedResolutionValue === '15' || selectedResolutionValue === '30') {
                         resolutionSelect.value = selectedResolutionValue + ' mins';
                    } else if (selectedResolutionValue === '60') {
                        resolutionSelect.value = '60 mins';
                    }
                }
                
                const refreshIntervalMs = {{ refresh_interval_ms }};
                const dataForm = document.getElementById('dataForm');
                
                if (refreshIntervalMs > 0) {
                    function autoRefresh() {
                        // Create a temporary input to trick Flask into thinking the request is a POST submission
                        const tempInput = document.createElement('input');
                        tempInput.type = 'hidden';
                        tempInput.name = 'auto_refresh';
                        tempInput.value = 'true';
                        dataForm.appendChild(tempInput);
                        dataForm.submit();
                    }

                    // Delay the first auto-refresh to prevent immediate double-post
                    setTimeout(function() {
                        setInterval(autoRefresh, refreshIntervalMs);
                    }, refreshIntervalMs);
                }
            </script>
          </body>
        </html>
    '''
    return render_template_string(
        template_html,
        plot_data=plot_data,
        oi_plot_data=oi_plot_data,
        input_values=input_values,
        refresh_interval_ms=refresh_interval_ms,
        get_chart_dpi=get_chart_dpi,
        cross_alert=cross_alert,
        expiry_data=expiry_data,
        latest_price_1=latest_price_1,
        latest_price_2=latest_price_2,
        total_call_oi=total_call_oi,
        total_put_oi=total_put_oi,
        # --- NEW VALUES PASSED TO TEMPLATE ---
        max_call_oi_strike=max_call_oi_strike,
        max_call_oi_value=max_call_oi_value,
        max_put_oi_strike=max_put_oi_strike,
        max_put_oi_value=max_put_oi_value,
        # --- END NEW VALUES ---
        now=datetime.datetime.now().strftime('%H:%M:%S')
    )

if __name__ == '__main__':
    # Setting debug to False when using 'Agg' backend in production is recommended,
    # but kept as True for development purposes.
    app.run(debug=True)
