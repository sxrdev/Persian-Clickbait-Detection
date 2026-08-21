import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix
from feature_database import DATABASE_HAND_FEATURES


class GET_IMPORTED_FEATURES(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.features_db = DATABASE_HAND_FEATURES()

        self.all_eghraghi = self.features_db.get_all_eghraghi_words()
        self.all_ehsasi = self.features_db.get_all_ehsasi_words()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            text = str(text)
            words = text.split()
            allcount = max(len(words), 1)
            
            searchedFeatures = []
            eghCount = sum(1 for word in self.all_eghraghi if word in text)
            searchedFeatures.append(eghCount / allcount)
            


            emoCount = sum(1 for word in self.all_ehsasi if word in text)
            searchedFeatures.append(emoCount / allcount)
            
            features.append(searchedFeatures)
        
        return np.array(features)


class COMB_MODEL_VECTOR_HAND_FEATURES(BaseEstimator, TransformerMixin):   

    def __init__(self, tfidf_vectorizer, by_hand_feat):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.by_hand_feat = by_hand_feat
    
    def fit(self, X, y=None):
        self.tfidf_vectorizer.fit(X)
        self.by_hand_feat.fit(X)
        return self
    
    def transform(self, X):
        # استخراج ویژگی‌های TF-IDF
        tfidf_features = self.tfidf_vectorizer.transform(X)
        
        # استخراج ویژگی‌های دستی
        by_hand_feat = self.by_hand_feat.transform(X)
        
        # ترکیب ویژگی‌ها
        manual_sparse = csr_matrix(by_hand_feat)
        combined_features = hstack([tfidf_features, manual_sparse])
        
        return combined_features