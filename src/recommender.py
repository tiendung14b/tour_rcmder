import numpy as np
import pandas as pd

class RecommenderSystem:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        
    def recommend_for_user(self, original_user_id, visiting_mode, target_city, top_k=5):
        """
        Gợi ý danh sách địa điểm cho một User dựa trên ngữ cảnh và lọc theo thành phố.
        
        Args:
            original_user_id (int): ID người dùng gốc.
            visiting_mode (str/int): Ngữ cảnh chuyến đi.
            target_city (str/int): Tên hoặc ID thành phố mục tiêu để hậu lọc (post-filtering).
            top_k (int): Số lượng địa điểm gợi ý.
            
        Returns:
            pd.DataFrame: Danh sách Top-K địa điểm phù hợp.
        """
        # 1. Transform User ID to internal index
        try:
            user_idx = self.data_loader.user_le.transform([original_user_id])[0]
        except ValueError:
            print(f"User {original_user_id} not found in training data.")
            return pd.DataFrame()
            
        # 1.5. Lấy danh sách Item ID mà User đã tương tác (Seen items)
        user_history = self.data_loader.transactions[self.data_loader.transactions['UserId'] == original_user_id]['AttractionId'].tolist()
        
        # 2. Get all item indices
        all_item_indices = np.arange(len(self.data_loader.item_le.classes_))
        
        # 3. Predict scores for all items
        item_features = self.data_loader.get_item_features()
        scores = self.model.predict_for_user(
            np.array([user_idx] * len(all_item_indices)),
            all_item_indices,
            item_features=item_features
        )
        
        # 4. Map indices back to original IDs and associate with scores
        original_item_ids = self.data_loader.item_le.inverse_transform(all_item_indices)
        
        predictions = pd.DataFrame({
            'AttractionId': original_item_ids,
            'Score': scores
        })
        
        # 5. Get Item details for Post-filtering
        item_details = self.data_loader.get_item_info(original_item_ids)
        merged_preds = pd.merge(predictions, item_details, on='AttractionId')
        
        # 6. Hậu lọc (Post-filtering) theo thành phố
        # Kiểm tra nếu target_city là string (tên) hay số (ID)
        if isinstance(target_city, str):
            filtered = merged_preds[merged_preds['CityName'].str.lower() == target_city.lower()]
        else:
            filtered = merged_preds[merged_preds['CityId'] == target_city]
            
        # (Optional) Hậu lọc theo Visiting Mode nếu cần thiết, 
        # nhưng ở baseline này ta chủ yếu lọc theo target_city.
            
        # 6.5. Filter out seen items (Unseen only)
        filtered = filtered[~filtered['AttractionId'].isin(user_history)]
            
        # 7. Sắp xếp giảm dần theo điểm dự đoán và lấy Top-K
        top_items = filtered.sort_values(by='Score', ascending=False).head(top_k)
        
        # Return format: Tên địa điểm, Thể loại, Điểm dự đoán và Thành phố
        result = top_items[['Attraction', 'AttractionTypeId', 'Score', 'CityName']].copy()
        
        # (Tùy chọn) Merge tên loại địa điểm nếu Load Type.xlsx
        # Hiện tại trả về AttractionTypeId
        return result
