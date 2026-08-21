import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dataset import Bad_Titles, Good_Titles
from features import GET_IMPORTED_FEATURES, COMB_MODEL_VECTOR_HAND_FEATURES
import matplotlib.pyplot as plt


all_titles = Bad_Titles + Good_Titles
labels = [1] * len(Bad_Titles) + [0] * len(Good_Titles)  #لیبل گذاری


tfidf = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2),
    analyzer='word',
    sublinear_tf=True,
    min_df=2,
    max_df=0.9
)

by_hand_feat = GET_IMPORTED_FEATURES()
combined = COMB_MODEL_VECTOR_HAND_FEATURES(tfidf, by_hand_feat)
X = combined.fit_transform(all_titles)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0, stratify=y )

model = LogisticRegression(
    C=0.3,
    max_iter=2000,
    class_weight='balanced',
    solver='liblinear',
    # penalty='l2'
    random_state=0
)

pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100}")

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision * 100}")

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall * 100}")

f1 = f1_score(y_test, y_pred)
print(f"f1 score: {f1 * 100}")




from sklearn.model_selection import cross_val_score

# فقط دقت
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print(f"\nCross-Validation (5-Fold):")
print(f"   Fold 1: {cv_scores[0]*100}")
print(f"   Fold 2: {cv_scores[1]*100}")
print(f"   Fold 3: {cv_scores[2]*100}")
print(f"   Fold 4: {cv_scores[3]*100}")
print(f"   Fold 5: {cv_scores[4]*100}")
print(f"   ─────────────────────")
print(f"   میانگین: {cv_scores.mean()*100}")
print(f"   انحراف معیار: ±{cv_scores.std()*100}")


y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc})")
plt.plot([0, 1], [0, 1], linestyle="--") 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()



precision, recall, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)
plt.figure()
plt.plot(recall, precision,label=f'Precision-Recall (AP = {avg_precision:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.show()


train_sizes, train_scores, test_scores = learning_curve(pipeline, X_train, y_train)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.figure()
plt.plot(train_sizes, train_mean, label='Training Score')
plt.plot(train_sizes, test_mean, label='Cross-Validation Score')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curv')
plt.legend()
plt.show()



with open('train_model/model_trained.pkl', 'wb') as a:
    pickle.dump(pipeline, a)

with open('train_model/comb_hand_and_model_features.pkl', 'wb') as a:
    pickle.dump(combined, a)
