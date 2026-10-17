import pandas as pd
import numpy as np

# 1. Sinh dữ liệu giả lập (Giữ nguyên)
def generate_synthetic_data (seed: int = 42):
    np.random.seed(seed)
    n_customers = 1000
    customers_data = {
        'customer_id': [f"CUST_{i:04d}" for i in range(1, n_customers + 1)],
        'customer_name': [f"Customer_{i}" for i in range(1, n_customers + 1)],
        'region': np.random.choice(['North', 'South', 'Central'], size=n_customers, p=[0.4, 0.4, 0.2]),
        'signup_date': pd.date_range(start='2024-01-01', periods=n_customers, freq='D')
    }
    df_customers = pd.DataFrame(customers_data)

    n_orders = 10000
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='h')
    orders_data = {
        'order_id': [f"ORD_{i:06d}" for i in range(1, n_orders + 1)],
        'customer_id': np.random.choice(df_customers['customer_id'], size=n_orders),
        'order_date': np.random.choice(dates, size=n_orders),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home & Kitchen', 'Books'], size=n_orders),
        'price': np.random.uniform(10.0, 500.0, size=n_orders).round(2),
        'quantity': np.random.randint(1, 6, size=n_orders),
        'discount': np.random.choice([0.0, 0.05, 0.1, 0.2, np.nan], size=n_orders, p=[0.5, 0.2, 0.15, 0.05, 0.1]),
        'status': np.random.choice(['COMPLETED', 'CANCELLED', 'REFUNDED'], size=n_orders, p=[0.8, 0.15, 0.05])
    }
    df_orders = pd.DataFrame(orders_data)

    return df_orders, df_customers

class DataUtils:
    def __init__(self, df_orders: pd.DataFrame, df_customers: pd.DataFrame = None):
        self.df = df_orders.copy()
        self.df_customers = df_customers.copy() if df_customers is not None else None
        self.df_completed = None
        self.df_merged = None

    #Inspecting
    def get_info(self, n_rows: int = 5) -> "DataUtils":
        print(n_rows)
        print(self.df.head(n_rows))
        self.df.info()
        return self

    def get_describe(self) :
        return self.df.describe()

    #  Null values
    def fill_null_values(self, column: str, value: float = 0.0) -> "DataUtils":
        self.df[column].fillna(value, inplace=True)
        return self

    # Feature engineering
    def create_total_amount(self) -> "DataUtils":
        self.df['total_amount'] = self.df['price'] * self.df['quantity'] * (1 - self.df['discount'].fillna(0))
        return self

    #Filtering completed orders
    def filter_completed_orders(self) -> "DataUtils":
        self.df_completed = self.df[self.df['status'] == 'COMPLETED'].copy()
        return self

    #  Category Analysis
    def category_amount(self) -> pd.DataFrame:
        if self.df_completed is None:
            self.filter_completed_orders()

        category_stats = self.df_completed.groupby('category').agg(
            total_revenue=('total_amount', 'sum'),
            total_quantity=('quantity', 'sum')
        ).reset_index().sort_values(by='total_revenue', ascending=False)
        
        return category_stats

    #  Monthly Revenue Analysis
    def create_monthly_revenue(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.df_completed is None:
            self.filter_completed_orders()

        self.df_completed['order_month'] = pd.to_datetime(self.df_completed['order_date']).dt.to_period('M')
        monthly_stats = self.df_completed.groupby('order_month')['total_amount'].sum().reset_index()
        top_month = monthly_stats.loc[monthly_stats['total_amount'].idxmax()]
        
        return monthly_stats, top_month

    #  Join df_completed với df_customers
    def merge_with_customers(self, how: str = 'inner') -> "DataUtils":
        if self.df_completed is None:
            self.filter_completed_orders()

        if self.df_customers is None:
            raise ValueError("Chưa truyền df_customers vào DataUtils!")

        self.df_merged = pd.merge(
            self.df_completed,
            self.df_customers,
            on='customer_id',
            how=how
        )
        return self

    #  Phân tích hiệu suất theo Vùng miền
    def regional_performance(self) -> pd.DataFrame:
        if self.df_merged is None:
            self.merge_with_customers()

        return self.df_merged.groupby('region').agg(
            total_revenue=('total_amount', 'sum'),
            completed_orders=('order_id', 'count')
        ).reset_index().sort_values(by='total_revenue', ascending=False)

    #  Top VIP Customers
    def get_top_vip_customers(self, top_n: int = 5) -> pd.DataFrame:
        if self.df_merged is None:
            self.merge_with_customers()

        return self.df_merged.groupby(['customer_id', 'customer_name', 'region']).agg(
            total_revenue=('total_amount', 'sum'),
            total_orders=('order_id', 'count')
        ).reset_index().sort_values(by='total_revenue', ascending=False).head(top_n)

    #  Xuất file Parquet
    def export_to_parquet(self, df: pd.DataFrame, filename: str = "top_vip_customers.parquet") -> None:
        df.to_parquet(filename, index=False)
        print(f" Đã xuất thành công file: {filename}")