from Utils import generate_synthetic_data, DataUtils
def run_with_dy():

    # 0. Sinh dữ liệu
    df_orders, df_customers = generate_synthetic_data(seed=42)

    # Khởi tạo class DataUtils với cả 2 DataFrame
    pipeline = DataUtils(df_orders, df_customers)

    # Câu 1, 2, 3, 4: Method Chaining
    pipeline.get_info() \
            .fill_null_values('discount', 0.0) \
            .create_total_amount() \
            .filter_completed_orders()

    # Câu 5: Category Analysis
    print("CÂU 5: Doanh thu theo Category", pipeline.category_amount())

    # Câu 6: Monthly Revenue
    print("CÂU 6: Tháng doanh revenue cao nhất ")
    monthly_df, top_month = pipeline.create_monthly_revenue()
    print(monthly_df)
    print("Tháng cao nhất: {top_month['order_month']} ({top_month['total_amount']:,.2f})")

    # Câu 7, 8: Merge & Region Performance
    pipeline.merge_with_customers(how='inner')
    print("CÂU 8: Phân tích theo Region", pipeline.regional_performance())

    # Câu 9: Top VIP Customer
    print("CÂU 9: Top 5 VIP Customer")
    top_vip = pipeline.get_top_vip_customers(top_n=5)
    print(top_vip.to_string(index=False))

    # Câu 10: Export File Parquet
    print("CÂU 10: Xuất file Parquet ")
    pipeline.export_to_parquet(top_vip, "top_vip_dynamic.parquet")


if __name__ == "__main__":
    run_with_dy()