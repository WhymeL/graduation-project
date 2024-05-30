import financedatabase as fd
from financetoolkit import Toolkit

# equities = fd.Equities()

# data = equities.select(country='China')
# data.to_excel("china_equities.xlsx")

# start_date = 2014-05-22
companies = Toolkit(
    # JD, ACGBY, HCSG, AAPL
    tickers=['JD', 'ACGBY', 'HCSG', 'AAPL'],
    api_key='Ywxs8CGJt7nWmke7VjRD3MJXyqost7yJ'
)
historical_data = companies.get_historical_data()
historical_data.to_excel("historical_data.xlsx")