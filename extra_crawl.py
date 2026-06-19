import json
import requests
import pandas as pd
import os
import time
from tqdm import tqdm
import concurrent.futures

# ==========================================
API_ENDPOINT = 'https://realtime.oxylabs.io/v1/queries'

# Cập nhật tài khoản mới vào đây:
API_AUTH = ('hungle0006_wv4Oc', 'Hungleduc006__')

# ==========================================
# 2. ĐƯỜNG DẪN FILE ĐỌC/GHI
# ==========================================
output_dir = os.path.expanduser("~/workspace/project_ml/")
# Đọc file cào được từ script Phase 1 của mày
BASIC_FILE = os.path.join(output_dir, "zillow_mass_dataset_full.csv") 

# Khai báo file lưu Extra
EXTRA_FILE = os.path.join(output_dir, "zillow_extra_features_only.csv")
CHECKPOINT_FILE = os.path.join(output_dir, "zillow_extra_checkpoint.csv")

# ==========================================
# 3. HÀM CÀO 1 LINK (BÓC TÁCH JSON THÔNG MINH)
# ==========================================
def fetch_extra_for_one_house(url):
    """
    Nhận vào 1 URL nhà, dùng Oxylabs tải trang và lôi cái ruột JSON __NEXT_DATA__ ra.
    Trả về 1 dictionary chứa các Features để ghép vào model ML.
    """
    if not isinstance(url, str):
        return None

    # Zillow đôi khi trả URL bị thiếu domain
    if url.startswith("/"):
        url = "https://www.zillow.com" + url

    payload = {
        "source": "universal",
        "url": url,
        "user_agent_type": "desktop",
        "render": "html"
    }

    dic = {"url": url} # Lưu url để lát sau mapping (Join) lại với file Basic

    try:
        res = requests.post(API_ENDPOINT, json=payload, auth=API_AUTH)
        if res.status_code == 200:
            html_content = res.json()["results"][0]["content"]
            
            # ZILLOW 2024/2025: Toàn bộ data Extra nằm trong cục JSON khổng lồ này
            if "id=\"__NEXT_DATA__\"" in html_content:
                start_str = 'id="__NEXT_DATA__" type="application/json">'
                end_idx = html_content.find('</script>', html_content.find(start_str))
                json_str = html_content[html_content.find(start_str) + len(start_str):end_idx]
                
                full_json = json.loads(json_str)
                
                # Men theo đường dẫn JSON để lấy dữ liệu Fact & Features
                cache = full_json.get("props", {}).get("pageProps", {}).get("componentProps", {}).get("gdpClientCache", {})
                if cache:
                    cache_key = list(cache.keys())[0]
                    reso = cache[cache_key].get("property", {}).get("resoFacts", {})
                    
                    # Mô phỏng lại chính xác định dạng của file mẫu (Dictionary/List)
                    # Bài mẫu yêu cầu các cột này, tao map trực tiếp luôn khỏi cào text
                    dic["Bedrooms and bathrooms"] = {
                        "Bedrooms": reso.get("bedrooms"),
                        "Bathrooms": reso.get("bathrooms"),
                        "Full bathrooms": reso.get("bathroomsFull"),
                        "1/2 bathrooms": reso.get("bathroomsHalf")
                    }
                    dic["Heating"] = {"Heating features": reso.get("heating")}
                    dic["Cooling"] = {"Cooling features": reso.get("cooling")}
                    dic["Appliances"] = {"Appliances included": reso.get("appliances")}
                    dic["Parking"] = {
                        "Parking features": reso.get("parkingFeatures"),
                        "Garage spaces": reso.get("garageSpaces"),
                        "Total spaces": reso.get("parking")
                    }
                    dic["Type and style"] = {
                        "Property subType": reso.get("propertySubType"),
                        "Architectural style": reso.get("architecturalStyle")
                    }
                    dic["Material information"] = {
                        "Construction materials": reso.get("constructionMaterials"),
                        "Roof": reso.get("roof"),
                        "Foundation": reso.get("foundation")
                    }
                    dic["Lot"] = {"Lot features": reso.get("lotFeatures")}
                    dic["Condition"] = {"Year built": reso.get("yearBuilt")}
                    dic["Utility"] = {
                        "Sewer information": reso.get("sewer"),
                        "Water information": reso.get("waterSource")
                    }
    except Exception as e:
        pass # Kệ lỗi, trả về dic rỗng cho các giá trị Extra
        
    return dic

# ==========================================
# 4. TRỤC CHÍNH (MULTITHREADING)
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(BASIC_FILE):
        print(f"Không tìm thấy file {BASIC_FILE} để lấy link.")
        exit()

    df_basic = pd.read_csv(BASIC_FILE)
    
    # Tìm xem cái cột chứa Link nó tên là gì (thường là detailUrl)
    url_col = 'detailUrl' if 'detailUrl' in df_basic.columns else [c for c in df_basic.columns if 'detailUrl' in c][0]
    all_urls = df_basic[url_col].dropna().tolist()

    print(f"Tìm thấy {len(all_urls)} links. Bắt đầu Phase 2 cào Extra Features...")
    
    extra_data_list = []
    MAX_WORKERS = 5 # Đặt 5 luồng để không bị Zillow sút (ban)
    
    # Chạy đa luồng bằng ThreadPoolExecutor cho bốc
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        futures = {executor.submit(fetch_extra_for_one_house, url): url for url in all_urls}
        
        # Lấy kết quả với thanh tiến trình TQDM
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(all_urls), desc="Scraping Extra")):
            res_dict = future.result()
            if res_dict and len(res_dict) > 1: # Đảm bảo có data ngoài cái url
                # Chuyển dictionary thành string chuẩn form file mẫu của mày
                for k, v in res_dict.items():
                    if k != "url" and v: 
                        # Format list string y như bài mẫu: "['Heating features: Hot Water', ...]"
                        formatted_list = [f"{sub_k}: {sub_v}" for sub_k, sub_v in v.items() if sub_v is not None]
                        res_dict[k] = str(formatted_list)
                extra_data_list.append(res_dict)
                
            # Auto-save mỗi 200 căn
            if (i + 1) % 200 == 0:
                pd.DataFrame(extra_data_list).to_csv(CHECKPOINT_FILE, index=False, encoding='utf-8-sig')

    # Lưu kết quả cuối cùng
    df_extra = pd.DataFrame(extra_data_list)
    df_extra.to_csv(EXTRA_FILE, index=False, encoding='utf-8-sig')
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        
    print(f"\n[THÀNH CÔNG] Đã cào xong Phase 2. File Extra nằm tại:\n-> {EXTRA_FILE}")