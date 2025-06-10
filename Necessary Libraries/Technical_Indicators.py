# Importing libraries
import numpy as np
import pandas as pd

class TI:

	def __init__(self, ohlcv_df):
		"""
		Initialise class with OHLCV parameters
		"""

		self.open = ohlcv_df['Open']
		self.high = ohlcv_df['High']
		self.low = ohlcv_df['Low']
		self.close = ohlcv_df['Close']
		self.volume = ohlcv_df['Volume']

	# Function that returns 0 for extremely small x and x otherwise
	def zero(self,x):
		return 0 if abs(x) < np.finfo(float).eps else x

	# Function that returns the difference of two series and adds epsilon to any zero values so we avoid dividing by 0
	def nonzero_sub(self,x,y):
		diff = x - y
		if diff.eq(0).any().any():
			diff += np.finfo(float).eps
		return diff

	# Function that returns the difference of two series and adds epsilon to any zero values so we avoid dividing by 0
	def nonzero_sum(self,x,y):
		sum_ = x + y
		if sum_.eq(0).any().any():
			sum_ += np.finfo(float).eps
		return sum_
		
	def sma(self, length, data=None):
		"""
		Function to calculate SMA

		Parameters:
		- length: int, window size for the SMA
		- data: pd.Series, price series to calculate SMA on
				if None, defaults to self.close
		
		Returns:
		- pd.Series of SMA values
		"""

		# If custom closing prices not given, take input as self.close
		if data is None:
			data = self.close

		# Simply use a rolling function to look at the past previous n days and then take the mean of the prices from these days
		return data.rolling(window=length).mean()

	def ema(self, length, data=None):
		"""
		Function to calculate EMA

		Parameters:
		- length: int, window size for the EMA
		- data: pd.Series, price series to calculate EMA on
				if None, defaults to self.close
		
		Returns:
		- pd.Series of EMA values
		"""

		# If custom closing prices not given, take input as self.close
		if data is None:
			data = self.close

		# Make a copy of our closing prices
		close = data.copy()
		
		# For our nth day, we calculate the SMA for reasons explained above
		sma_nth = close[0:length].mean()
		
		# Set previous values to NaN and set nth day to the calculated value
		close[:length-1] = np.nan
		close.iloc[length-1] = sma_nth
		
		# Calculate the EMA for the rest of the days using .ewm function and taking mean
		return close.ewm(span=length, adjust=False).mean()

	def rma(self,length, data=None):
		"""
		Function to calculate RMA

		Parameters:
		- length: int, window size for the RMA
		- data: pd.Series, price series to calculate RMA on
				if None, defaults to self.close
		
		Returns:
		- pd.Series of RMA values
		"""

		# If custom closing prices not given, take input as self.close
		if data is None:
			data = self.close

		# We can use the same ewm function as for our EMA function
		# But we add a specified smoothing factor of 1/length
		return data.ewm(min_periods = length, alpha= 1/length).mean()

	def macd(self, data=None):
		"""
		Function to calculate RMA

		Parameters:
		- length: int, window size for the RMA
		- data: pd.Series, price series to calculate RMA on
				if None, defaults to self.close
		
		Returns:
		- pd.Series of RMA values
		"""

		# If custom closing prices not given, take input as self.close
		if data is None:
			data = self.close

		# Calculate the 12 and 26 day EMAs
		fast = self.ema(length=12)
		slow = self.ema(length=26)
		
		# Calculate MACD
		macd = fast - slow
		
		# Calculate Signal and Histogram
		signal = self.ema(length=9, data=macd.loc[macd.first_valid_index():,])
		hist = macd - signal
		
		# Put values into dataframe to easily call each one
		df = pd.DataFrame(data={"macd": macd, "signal": signal, "hist": hist})
		return df

	def atr(self, length, data=None):
		"""
		Function to calculate ATR

		Parameters:
		- length: int, window size for the ATR
		- data: pd.DataFrame, with 'High', 'Low', 'Close' columns
				if None, defaults to initialised prices
		
		Returns:
		- pd.Series of ATR values
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			high = self.high
			low = self.low
			close = self.close
		else:
			high = data['High']
			low = data['Low']
			close = data['Close']
		
		# Calculate True Ranges
		tr0 = abs(high - low)
		tr0.iloc[0] = np.nan
		tr1 = abs(high - close.shift())
		tr2 = abs(low - close.shift())
		tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
		
		# Calculate ATR
		atr = self.rma(length, data=tr)
		return atr

	def adx(self, length, data=None):
		"""
		Function to calculate ADX

		Parameters:
		- length: int, window size for the ADX
		- data: pd.DataFrame, with 'High', 'Low', 'Close' columns
				if None, defaults to initialised prices
		
		Returns:
		- pd.DataFrame of ADX values, PDIs and NDIs
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			high = self.high
			low = self.low
			close = self.close
		else:
			high = data['High']
			low = data['Low']
			close = data['Close']

		# Calculate ATR values with a period of n days
		atr = self.atr(length, data=data)
		
		# Find differences in High and Low prices on consecutive days
		high_ = high - high.shift()
		low_ = low.shift() - low
		
		# Find Positive and Negative Directional Movements
		pdm = ((high_ > low_) & (high_ > 0)) * high_
		ndm = ((low_ > high_) & (low_ > 0)) * low_
		
		# Apply our zero function for extremely small values
		pdm = pdm.apply(self.zero)
		ndm = ndm.apply(self.zero)
		
		# Smooth Positive and Negative Directional Movements using RMAs
		k = 100 / atr
		pdi = k * self.rma(length, data=pdm)
		ndi = k * self.rma(length, data=ndm)
		
		# Calculate ADX, the smoothed difference between +DI and -DI
		dx = 100 * (pdi - ndi).abs() / self.nonzero_sum(pdi,ndi)
		adx = self.rma(length, data=dx)
		
		# Store values into dataframe
		df = pd.DataFrame(data={"adx": adx, "pdi": pdi, "ndi": ndi})
		return df

	def psar(self, data=None):
		"""
		Function to calculate PSAR

		Parameters:
		- data: pd.DataFrame, with 'High', 'Low', 'Close' columns
				if None, defaults to initialised prices
		
		Returns:
		- pd.DataFrame of PSAR Long and Short values, Acceleration Factors and Buy/Sell Signals
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			high = self.high
			low = self.low
			close = self.close
		else:
			high = data['High']
			low = data['Low']
			close = data['Close']

		def falling(high,low):
			"""
			Define falling function that checks if the market is in a falling state by checking if the downward movement (down) is greater than the upward movement (up), and if down is positive.
			We return true if this is the case and false if not.
			"""
			up = high - high.shift()
			down = low.shift() - low
			dm = (((down > up) & (down > 0)) * down).apply(self.zero).iloc[-1]
			return dm > 0
		
		# Assign values to af0, af and max_af
		af0 = 0.02
		af = 0.02
		max_af = 0.2
		
		# Determine the trend based on the first two data points
		fall = falling(high.iloc[:2], low.iloc[:2])
		
		# The starting value of SAR is initialized to the first adjusted close price
		sar = close.iloc[0]
		
		# If the market is falling (fall == True), set the Extreme Point to the first low price and if the market is not falling, set it to the first high price
		if fall:
			ep = low.iloc[0]
		else:
			ep = high.iloc[0]
		"""
		Initialise arrays that will store:
		- Parabolic SAR values for long and short positions
		- Array to store the acceleration factor values for each step, starting with the initial acceleration factor (af0) for the first two data points
		- Array with 0s to track reversal points
		"""
		long = pd.Series(np.nan, index=high.index)
		short = long.copy()
		af_array = long.copy()
		af_array.iloc[0:2] = af0
		reverse_array = pd.Series(0, index=high.index)
		
		#We calculate values for each of the 4 arrays above for each time step.
		#The number of time steps is the number of rows in our dataset.
		for i in range(1, len(close)):
			
			# Get the high and low prices for our particular time step
			high_ = high.iloc[i]
			low_ = low.iloc[i]
			
			# If we are in a falling position we calculate SAR and if a trend reversal is occuring
			if fall:
				sar1 = sar + af * (ep - sar)
				reverse = high_ > sar1
				
			# If the current low price is below the extreme point, we update EP to the current low and increase the acceleration factor, ensuring it does not exceed the maximum value
				if low_ < ep:
					ep = low_
					af = min(af + af0, max_af)
					
			# Adjust SAR to be the maximum of the previous high prices and the current SAR value, ensuring that the SAR follows the trend
				sar1 = max(high.iloc[i - 1], high.iloc[i - 2], sar1)
				
			# Now we consider the case where we are not in a falling position
			else:
				sar1 = sar + af * (ep - sar)
				reverse = low_ < sar1
				
			# If the current high price is above the extreme point, we update EP to the current high and increase the acceleration factor, ensuring it does not exceed the maximum value
				if high_ > ep:
					ep = high_
					af = min(af + af0, max_af)
					
			# Adjust SAR to be the minimum of the previous low prices and the current SAR value, ensuring that the SAR follows the trend
				sar1 = min(low.iloc[i - 1], low.iloc[i - 2], sar1)
			"""
			If a trend reversal is detected:
			- SAR is reset to the extreme point
			- Acceleration factor is reset to af0, and the direction of the trend (fall) is flipped (from falling to rising, or vice versa)
			- Extreme point is set to the new low or high depending on the new trend direction
			- If also in fall, then take negative so we can identify difference between buy and sell signals
			"""
			if reverse:
				sar1 = ep
				af = af0
				fall = not fall
				ep = low_ if fall else high_
				reverse = int(reverse) if fall else -int(reverse)
			
			# Update SAR
			sar = sar1

			# Depending on whether the trend is falling or rising, the SAR is added to the short or long array
			if fall:
				short.iloc[i] = sar
			else:
				long.iloc[i] = sar
				
			# Acceleration factor and reversal status are updated
			af_array.iloc[i] = af
			reverse_array.iloc[i] = int(reverse)
			
		# Store arrays into Dataframe and return it
		psardf = pd.DataFrame({"PSAR_Long": long,"PSAR_Short": short,"PSAR_af": af_array,"PSAR_Rev": reverse_array})
		return psardf

	def mfi(self, length, data=None):
		"""
		Function to calculate MFI

		Parameters:
		- length: int, window size for the MFI
		- data: pd.DataFrame, with 'High', 'Low', 'Close', 'Volume' columns
				if None, defaults to initialised prices/volumes
		
		Returns:
		- pd.Series of MFI values
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			high = self.high
			low = self.low
			close = self.close
			volume = self.volume
		else:
			high = data['High']
			low = data['Low']
			close = data['Close']
			volume = data['Volume']

		# Calculate Typical Price
		tp = (high + low + close)/3
		
		# Calculate Relative Money Flow
		rmf = tp * volume
		
		# Create arrays of 0s to store Positive and Negative Money Flows
		pmf = pd.Series(0.0, index=high.index)
		nmf = pmf.copy()
		
		# Assign RMF values for each depending on if today's TP is lower or higher than yesterday's TP
		pmf[tp.diff() > 0] = rmf[tp.diff() > 0]
		nmf[tp.diff() < 0] = rmf[tp.diff() < 0]
		
		# Sum PMF and NMF across a rolling window
		psum = pmf.rolling(window=length).sum()
		nsum = nmf.rolling(window=length).sum()
		
		# Calculate MFI
		mfi = 100 * psum / self.nonzero_sum(psum,nsum)
		return mfi

	def so(self, length, data=None):
		"""
		Function to calculate SO

		Parameters:
		- length: int, window size for the SO
		- data: pd.DataFrame, with 'High', 'Low', 'Close' columns
				if None, defaults to initialised prices
		
		Returns:
		- pd.DataFrame of SO Fast (k) and Slow (d) values
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			high = self.high
			low = self.low
			close = self.close
		else:
			high = data['High']
			low = data['Low']
			close = data['Close']

		# Calculate the lowest Low price and highest High price within a period of n days
		LL = low.rolling(length).min()
		HH = high.rolling(length).max()
		
		# We then calculate our SO_K before smoothing whilst making sure we avoid dividing by 0
		stoch = 100 * (close - LL)
		stoch /= self.nonzero_sub(HH, LL)
		
		# Calculate SO_K and SO_D
		sok = self.sma(length=3, data=stoch.loc[stoch.first_valid_index():,])
		sod = self.sma(length=3, data=sok.loc[sok.first_valid_index():,])
		
		# Put both arrays into a dataframe and return it
		results = pd.DataFrame(data={"so_k": sok, "so_d": sod})
		return results

	def rsi(self, length, data=None):
		"""
		Function to calculate RSI

		Parameters:
		- length: int, window size for the RSI
		- data: pd.Series, closing price series to calculate RSI on
				if None, defaults to self.close
		
		Returns:
		- pd.Series of RSI values
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			data = self.close

		# Calculate difference in each time step of closing prices
		data = data.diff()
		
		# First make a copy of this twice
		up_df, down_df = data.copy(), data.copy()

		# For up days, if the change is less than 0 set to 0
		up_df[up_df < 0] = 0
		# For down days, if the change is greater than 0 set to 0
		down_df[down_df > 0] = 0
		# We need change in price to be positive
		down_df = down_df.abs()

		# Calculate the RMA of the up and down days
		rma_up = self.rma(length=length, data=up_df)
		rma_down = self.rma(length=length, data=down_df)

		# Calculate the Relative Strength
		RS = rma_up / rma_down

		# Calculate the Relative Strength Index
		RSI = 100.0 - (100.0 / (1.0 + RS))
		return RSI

	def srsi(self, length, data=None):
		"""
		Function to calculate SRSI

		Parameters:
		- length: int, window size for the SRSI
		- data: pd.Series, closing price series to calculate SRSI on
				if None, defaults to self.close
		
		Returns:
		- pd.Series of SRSI values
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			data = self.close

		# Calculate the RSI using our closing prices using our previously defined function
		rsi1 = self.rsi(length=length, data=data)
		
		# Then find the max and min RSI over a window
		rsi_min = rsi1.rolling(length).min()
		rsi_max = rsi1.rolling(length).max()
		
		# Calculate SRSI before smoothing, being careful to avoid dividing by 0
		srsi = 100 * (rsi1 - rsi_min)
		srsi /= self.nonzero_sub(rsi_max, rsi_min)
		
		# Apply SMA smoothing with a window of 3 days
		srsi1 = self.sma(length=3,data=srsi)
		return srsi1

	def fi(self, length, data=None):
		"""
		Function to calculate FI

		Parameters:
		- length: int, window size for the FI
		- data: pd.DataFrame, with 'Close', 'Volume' columns
				if None, defaults to initialised prices/volumes
		
		Returns:
		- pd.Series of FI values
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			close = self.close
			volume = self.volume
		else:
			close = data['Close']
			volume = data['Volume']

		# Calculate our FI
		temp = close.diff() * volume
		fi1 = self.ema(length=length,data=temp)
		return fi1

	def obv(self, data=None):
		"""
		Function to calculate OBV

		Parameters:
		- data: pd.DataFrame, with 'Close', 'Volume' columns
				if None, defaults to initialised prices/volumes
		
		Returns:
		- pd.Series of OBV values
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			close = self.close
			volume = self.volume
		else:
			close = data['Close']
			volume = data['Volume']

		# Calculate the differences in Closing prices for consecutive days
		sign = close.diff()
		
		# Assign 1 or -1 to each stock on each day dependent on if the prices are going up or down respectively
		sign[sign > 0] = 1
		sign[sign < 0] = -1
		sign.iloc[0] = 1
		
		# Multiply by volume to either get a positive or negative volume dependent on if prices have gone up or down
		signed_volume = sign * volume
		
		# Take a cumulative sum to get the OBV for any given day
		obv = signed_volume.cumsum()
		return obv

	def std(self, length, data=None):
		"""
		Function to calculate STD

		Parameters:
		- length: int, window size for the STD
		- data: pd.Series, closing price series to calculate STD on
				if None, defaults to self.close
		
		Returns:
		- pd.Series of STD values
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			data = self.close

		# Calculate STD
		return data.rolling(length, min_periods=length).std(ddof=1)

	def bbands(self, length, data=None):
		"""
		Function to calculate BBands

		Parameters:
		- length: int, window size for the BBands
		- data: pd.Series, closing price series to calculate BBands on
				if None, defaults to self.close
		
		Returns:
		- pd.DataFrame of BBand Low, Mid, Upper, Bandwidth and Percent
		"""

		# If custom prices not given, take input as initialised prices
		if data is None:
			data = self.close
			
		# Calculate our SMA to find BB_Mid
		mid = self.sma(length=length,data=data)
		
		# Calculate the Standard Deviation across a window to find BB_Low and BB_High
		bband_std = data.rolling(window=length).std(ddof=0)
		low = mid - 2*bband_std
		high = mid + 2*bband_std
		
		# Calculate Bollinger Bandwidth and Percent
		bandwidth = 100 * (high - low) / mid
		percent = (data - low) / (high - low)
		
		# Put arrays into dataframe and return it
		results = pd.DataFrame(data={"lower": low, "mid": mid, "upper": high, "bandwith": bandwidth, "percent": percent})
		return results