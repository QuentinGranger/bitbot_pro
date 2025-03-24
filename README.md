# BitBot Pro

## Overview
BitBot Pro is an open-source Bitcoin microtrading bot designed to provide hedge fund-like performance while being optimized for Raspberry Pi and low-budget setups. The bot leverages advanced trading strategies including technical analysis, on-chain data, sentiment analysis, and multi-exchange arbitrage (Binance, Bybit, OKX).

## Key Features
- **Performance Optimized**: Built with "Raspberry Pi first" approach, focusing on execution speed and low latency
- **Multi-Strategy Framework**: Combines technical, on-chain, sentiment and arbitrage analysis
- **Signal Aggregator**: Intelligent weighting of signals based on market context and strategy performance
- **Risk Management Pro**: Dynamic stop adjustments, drawdown limitation, and real-time volatility adaptation
- **Exchange Integration**: Support for Binance, Bybit, and OKX with fallback mechanisms
- **Resilient Architecture**: Smart caching systems and dynamic fallbacks to handle API outages
- **Web Interface**: Lightweight, responsive dashboard for monitoring and control
- **Telegram Alerts**: Real-time notifications for trades and system status
- **Community Driven**: Shared strategies and continuous improvement through user feedback

## Requirements
- Python 3.x
- Compatible with ARM64 architecture (Raspberry Pi)
- Internet connection for API access

## Installation

### On Raspberry Pi
```bash
# Update package lists
sudo apt update

# Install required system dependencies
sudo apt install -y python3-pip python3-venv libatlas-base-dev git

# Clone the repository
git clone https://github.com/yourusername/BitBotPro.git
cd BitBotPro

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Configure your API keys (see Configuration section)
cp config.example.py config.py
nano config.py
```

### On Other Systems
Follow similar steps as above, adjusting the system dependencies as needed for your operating system.

## Configuration
Edit the `config.py` file to add your exchange API keys and customize bot settings:
- Exchange API credentials
- Trading pairs and strategies
- Risk management parameters
- Notification settings

## Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Start the bot
python main.py

# Start the web interface
python web_interface.py
```

## Project Structure
```
BitBotPro/
├── config.py                  # Configuration settings
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── strategies/                # Trading strategies
│   ├── technical.py           # Technical analysis
│   ├── onchain.py             # On-chain analysis
│   ├── sentiment.py           # Sentiment analysis
│   └── arbitrage.py           # Arbitrage strategies
├── exchanges/                 # Exchange integrations
│   ├── binance.py             # Binance API
│   ├── bybit.py               # Bybit API
│   └── okx.py                 # OKX API
├── signal_aggregator/         # Signal weighting and processing
├── risk_management/           # Risk management tools
├── web_interface/             # Web dashboard
└── utils/                     # Utility functions
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
Trading cryptocurrencies involves significant risk. This software is for educational purposes only. Use at your own risk.
