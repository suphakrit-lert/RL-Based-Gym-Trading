import numpy as np
from scipy.stats import mstats  # For geometric mean, used in Sharpe Ratio calculation

from .trading_env import TradingEnv, Actions, Positions

# TODO: How to find short andl long
class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    # def _process_data(self):
    #     prices = self.df.loc[:, 'Close'].to_numpy()

    #     prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
    #     prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

    #     diff = np.insert(np.diff(prices), 0, 0)
    #     signal_features = np.column_stack((prices, diff))

    #     return prices.astype(np.float32), signal_features.astype(np.float32)
    
    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        print("princs: ", prices)
        
        # Validate index - Ensure we have enough data for the window size and indicators
        assert self.frame_bound[0] > self.window_size, "Frame bound start should be greater than window size."
        
        df = self.df.loc[self.frame_bound[0]-self.window_size:self.frame_bound[1]].copy()

        # Calculate Moving Averages
        df['SMA'] = df['Close'].rolling(window=5).mean()  # Short-term moving average
        df['LMA'] = df['Close'].rolling(window=20).mean()  # Long-term moving average
        
        # Calculate RSI
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Handling NaN values that may appear due to the rolling and ewm functions
        df.fillna(0, inplace=True)
        
        # Select the indicators and the price difference to be used as features
        features = df[['Close', 'SMA', 'LMA', 'RSI', 'MACD', 'Signal_Line']]
        
        # Calculate price differences as an additional feature
        diff = np.insert(np.diff(df['Close'].to_numpy()), 0, 0)
        features['Price_Diff'] = diff
        
        prices = df['Close'].to_numpy().astype(np.float32)
        signal_features = features.to_numpy().astype(np.float32)
        print("signal_features: ", signal_features)
        
        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

        return step_reward
    
    # TODO: Fix negative rewards
    # def _calculate_reward(self, action):
    #     trade = False
    #     if (action == Actions.Buy.value and self._position == Positions.Short) or \
    #     (action == Actions.Sell.value and self._position == Positions.Long):
    #         trade = True

    #     current_price = self.prices[self._current_tick]
    #     last_trade_price = self.prices[self._last_trade_tick]
    #     price_diff = current_price - last_trade_price

    #     step_reward = 0
    #     trade_cost = 0
    #     if self._position == Positions.Long:
    #         trade_cost = self.trade_fee_ask_percent * last_trade_price
    #         step_reward += price_diff - trade_cost  # profit from price increase
    #     elif self._position == Positions.Short:
    #         trade_cost = self.trade_fee_bid_percent * last_trade_price
    #         step_reward += -price_diff - trade_cost  # profit from price decrease

    #     # Adjust for Sharpe Ratio over a rolling window of returns
    #     window_size = 100
    #     if len(self.history['total_profit']) > window_size: 
    #         window_returns = np.diff(self.history['total_profit'][-window_size:]) / self.history['total_profit'][-window_size-1:-1]
    #         sharpe_ratio = np.mean(window_returns) / np.std(window_returns) if np.std(window_returns) != 0 else 0
    #         step_reward += sharpe_ratio

    #     return step_reward

    def _update_profit(self, action):
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

    # def _update_profit(self, action):
    #     trade = False
    #     if (action == Actions.Buy.value and self._position == Positions.Short) or \
    #     (action == Actions.Sell.value and self._position == Positions.Long):
    #         trade = True

    #     if trade:
    #         current_price = self.prices[self._current_tick]
    #         last_trade_price = self.prices[self._last_trade_tick]

    #         if self._position == Positions.Long:
    #             trade_cost = self.trade_fee_bid_percent * current_price
    #             shares = self._total_profit / (last_trade_price + trade_cost)
    #             self._total_profit = shares * (current_price - trade_cost)

    # def _update_profit(self, action):
    #     trade = False
    #     if (action == Actions.Buy.value and self._position == Positions.Short) or \
    #     (action == Actions.Sell.value and self._position == Positions.Long):
    #         trade = True

    #     if trade:
    #         current_price = self.prices[self._current_tick]
    #         last_trade_price = self.prices[self._last_trade_tick]

    #         if self._position == Positions.Long:
    #             # Exiting a Long position: Sell high
    #             sell_cost = self.trade_fee_bid_percent * current_price  # Cost when selling
    #             shares_bought = self._total_profit / last_trade_price
    #             self._total_profit = (shares_bought * current_price) - (shares_bought * sell_cost)
            
    #         elif self._position == Positions.Short:
    #             # Exiting a Short position: Buy low to cover
    #             buy_cost = self.trade_fee_ask_percent * current_price  # Cost when buying to cover
    #             # Here we assume the 'shares' are 'shorted' at 'last_trade_price' and now 'covered' at 'current_price'
    #             # The profit from a short sale is: (Price at Short Sale - Price at Cover) * Number of Shares - Costs
    #             money_from_short = self._total_profit * (last_trade_price / current_price)  # Adjust profit based on price change
    #             cost_of_buy_to_cover = self._total_profit * buy_cost / current_price  # Cost to cover the short
    #             self._total_profit = money_from_short - cost_of_buy_to_cover

    # def _update_profit(self, action):
    #     trade = False
    #     if (action == Actions.Buy.value and self._position == Positions.Short) or \
    #     (action == Actions.Sell.value and self._position == Positions.Long):
    #         trade = True

    #     if trade:
    #         current_price = self.prices[self._current_tick]
    #         last_trade_price = self.prices[self._last_trade_tick]

    #         if self._position == Positions.Long:
    #             # Exiting a Long position: Sell high
    #             sell_cost = self.trade_fee_bid_percent * current_price  # Cost when selling
    #             shares_bought = self._total_profit / last_trade_price
    #             self._total_profit = (shares_bought * current_price) - (shares_bought * sell_cost)
            
    #         elif self._position == Positions.Short:
    #             # Exiting a Short position: Buy low to cover
    #             buy_cost = self.trade_fee_ask_percent * current_price  # Cost when buying to cover
    #             # Here we assume the 'shares' are 'shorted' at 'last_trade_price' and now 'covered' at 'current_price'
    #             # The profit from a short sale is: (Price at Short Sale - Price at Cover) * Number of Shares - Costs
    #             money_from_short = self._total_profit * (last_trade_price / current_price)  # Adjust profit based on price change
    #             cost_of_buy_to_cover = self._total_profit * buy_cost / current_price  # Cost to cover the short
    #             self._total_profit = money_from_short - cost_of_buy_to_cover

    # TODO: Fix negative rewards
    # def _update_profit(self, action):
    #     trade = False
    #     if (action == Actions.Buy.value and self._position == Positions.Short) or \
    #     (action == Actions.Sell.value and self._position == Positions.Long):
    #         trade = True
    
    #     if trade:
    #         current_price = self.prices[self._current_tick]
    #         last_trade_price = self.prices[self._last_trade_tick]

    #         if self._position == Positions.Long:
    #             # Exiting a Long position: Sell high
    #             sell_cost = self.trade_fee_bid_percent * current_price  # Cost when selling
    #             shares_bought = self._total_profit / last_trade_price
    #             self._total_profit = (shares_bought * current_price) - (shares_bought * sell_cost)

    #         elif self._position == Positions.Short:
    #             # Exiting a Short position: Buy low to cover
    #             buy_cost = self.trade_fee_ask_percent * current_price  # Cost when buying to cover
    #             shares_shorted = self._total_profit / last_trade_price
    #             # The correct profit calculation when closing a short position
    #             self._total_profit = (shares_shorted * (last_trade_price - current_price)) - (shares_shorted * buy_cost)

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
