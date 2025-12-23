import pandas as pd

# Đọc các tệp CSV
file_1 = pd.read_csv('D:/0_Daniel/HK5/final_1.csv')
file_2 = pd.read_csv('D:/0_Daniel/HK5/final_2.csv')
file_3 = pd.read_csv('D:/0_Daniel/HK5/hackdb.csv')

# Kiểm tra các cột trong các tệp
print("Columns in file_1:", file_1.columns)
print("Columns in file_2:", file_2.columns)
print("Columns in file_3:", file_3.columns)

# Lấy cột 'Domain' hoặc 'Hacked_URL' từ mỗi tệp, kiểm tra sự tồn tại của từng cột
if 'Domain' in file_1.columns and 'Hacked_URL' in file_1.columns:
    file_1 = file_1[['Domain', 'Hacked_URL']].melt(id_vars=[], value_vars=['Domain', 'Hacked_URL'], var_name='type', value_name='url')
else:
    file_1 = file_1[['Domain']].rename(columns={'Domain': 'url'})

if 'Domain' in file_2.columns and 'Hacked_URL' in file_2.columns:
    file_2 = file_2[['Domain', 'Hacked_URL']].melt(id_vars=[], value_vars=['Domain', 'Hacked_URL'], var_name='type', value_name='url')
else:
    file_2 = file_2[['Domain']].rename(columns={'Domain': 'url'})

if 'Hacked_URL' in file_3.columns:
    file_3 = file_3[['Hacked_URL']].rename(columns={'Hacked_URL': 'url'})
else:
    file_3 = file_3[['Domain']].rename(columns={'Domain': 'url'})

# Tạo cột 'id' cho từng file (số liên tục)
file_1['id'] = range(1, len(file_1) + 1)
file_2['id'] = range(len(file_1) + 1, len(file_1) + len(file_2) + 1)
file_3['id'] = range(len(file_1) + len(file_2) + 1, len(file_1) + len(file_2) + len(file_3) + 1)

# Gộp tất cả các tệp lại
merged_data = pd.concat([file_1[['id', 'url']], file_2[['id', 'url']], file_3[['id', 'url']]], ignore_index=True)

# Lưu kết quả vào file dataset.csv
merged_data.to_csv('D:/0_Daniel/HK5/dataset.csv', index=False)

print("Đã hoàn thành gộp các tệp CSV thành dataset.csv")
