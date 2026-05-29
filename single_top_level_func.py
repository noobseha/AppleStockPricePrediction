import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import os

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

def run_integrated_analysis(data_path):
    # ==========================================
    # PART 1: Preprocessing 
    # ==========================================
    print("\n" + "=" * 60)
    print("PART 1. PREPROCESSING")
    print("=" * 60)
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date').reset_index(drop=True)
    df = df[df['Date'].dt.year >= 2020].reset_index(drop=True)
    
    # 2. Dirty Data Processing
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset='Title', keep='last')
    df = df.drop(columns=['Dividends', 'Stock Splits', 'Title', 'Link', 'Source', 'gpt_summary'], errors='ignore')
    
    # 3. Sentiment Feature Engineering
    df['signed_sentiment_score'] = df.apply(
        lambda row: row['gpt_sentiment_score'] if row['gpt_sentiment_direction'] == 'Positive' else -row['gpt_sentiment_score'],
        axis=1
    )
    df = df.drop(columns=['gpt_positive_score', 'gpt_negative_score'], errors='ignore')
    
    # 4. Merge Rare Categories
    threshold = 10
    value_counts = df['gpt_event_type'].value_counts()
    rare_categories = value_counts[value_counts < threshold].index.tolist()
    df['gpt_event_type'] = df['gpt_event_type'].apply(lambda x: 'Other' if x in rare_categories else x)
    
    # 5. Technical Indicator Engineering
    df['Daily_Volatility'] = (df['High'] - df['Low']) / df['Open']
    df['Buy_Pressure'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Sell_Pressure'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Close_Change_Rate'] = df['Close'].pct_change()
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['Volume_Change_Rate'] = df['Volume'].pct_change()
    
    # 6. Target and Market Status (Bull/Bear)
    df['Date_only'] = df['Date'].dt.date
    daily_close = df.groupby('Date_only')['Close'].first().reset_index().sort_values('Date_only')
    daily_close['Target'] = np.where(daily_close['Close'].shift(-1) > daily_close['Close'], 1, 0)
    
    last_date = daily_close['Date_only'].iloc[-1]
    daily_close = daily_close.iloc[:-1]
    daily_close['MA_20'] = daily_close['Close'].rolling(window=20).mean()
    daily_close['Bull_Bear'] = np.where(daily_close['Close'] > daily_close['MA_20'], 1, 0)
    
    df = df[df['Date_only'] != last_date]
    df = df.merge(daily_close[['Date_only', 'Target', 'MA_20', 'Bull_Bear']], on='Date_only', how='left')
    df = df.dropna().reset_index(drop=True)
    
    # 7. Visualization (Save Images)
    plt.figure(figsize=(14, 6))
    daily_plot = df.groupby('Date_only')['Close'].first()
    plt.plot(daily_plot.index, daily_plot.values)
    plt.title("Apple Close Price Time Series")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("timeseries_plot.png")
    plt.close()
    
    plt.figure(figsize=(6, 5))
    target_counts = df.groupby('Date_only')['Target'].first().value_counts()
    plt.bar(['Down(0)', 'Up(1)'], target_counts.values)
    plt.title("Target Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("target_distribution.png")
    plt.close()
    
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()
    print("-> Preprocessing: 3 plots (timeseries, target_dist, heatmap) saved locally.")
    
    # 8. Feature Selection
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'MA_20',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ADX', 'ATR', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Position',
        'Daily_Volatility', 'Buy_Pressure', 'Sell_Pressure', 'Close_Change_Rate', 'Volume_Change_Rate',
        'gpt_sentiment_score', 'signed_sentiment_score', 'gpt_relevance_to_apple', 'gpt_importance_score',
        'gpt_event_type', 'gpt_sentiment_direction', 'Bull_Bear'
    ]
    df = df[['Date', 'Date_only'] + features + ['Target']]
    
    # 9. One-Hot Encoding
    df = pd.get_dummies(df, columns=['gpt_event_type', 'gpt_sentiment_direction'], drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    feature_columns_only = [c for c in df.columns if c not in ['Date', 'Date_only']]
    df = df.drop_duplicates(subset=feature_columns_only).reset_index(drop=True)
    
    # 10. Train/Test Split
    split_date = df['Date'].max() - pd.DateOffset(months=3)
    train_df = df[df['Date'] < split_date].copy().reset_index(drop=True)
    test_df = df[df['Date'] >= split_date].copy().reset_index(drop=True)
    
    # 11. Scaling and CSV Save
    exclude_columns = ['Date', 'Date_only', 'Target', 'Bull_Bear'] + [c for c in df.columns if 'gpt_event_type_' in c or 'gpt_sentiment_direction_' in c]
    scale_columns = [c for c in df.columns if c not in exclude_columns]
    
    standard_train_df, standard_test_df = train_df.copy(), test_df.copy()
    robust_train_df, robust_test_df = train_df.copy(), test_df.copy()
    
    standard_scaler = StandardScaler()
    standard_train_df[scale_columns] = standard_scaler.fit_transform(standard_train_df[scale_columns])
    standard_test_df[scale_columns] = standard_scaler.transform(standard_test_df[scale_columns])
    
    robust_scaler = RobustScaler()
    robust_train_df[scale_columns] = robust_scaler.fit_transform(robust_train_df[scale_columns])
    robust_test_df[scale_columns] = robust_scaler.transform(robust_test_df[scale_columns])
    
    df.drop(columns=['Date_only']).to_csv("baseline_dataset_final.csv", index=False)
    train_df.drop(columns=['Date_only']).to_csv("train_dataset_raw.csv", index=False)
    test_df.drop(columns=['Date_only']).to_csv("test_dataset_raw.csv", index=False)
    standard_train_df.drop(columns=['Date_only']).to_csv("train_dataset_standard_scaled.csv", index=False)
    standard_test_df.drop(columns=['Date_only']).to_csv("test_dataset_standard_scaled.csv", index=False)
    robust_train_df.drop(columns=['Date_only']).to_csv("train_dataset_robust_scaled.csv", index=False)
    robust_test_df.drop(columns=['Date_only']).to_csv("test_dataset_robust_scaled.csv", index=False)
    print("-> Preprocessing: 7 dataset CSV files saved locally.")
    
    
    # ==========================================
    # PART 2: DS_TP_Regression
    # ==========================================
    print("\n" + "=" * 60)
    print("PART 2. REGRESSION ANALYSIS")
    print("=" * 60)
    
    def plot_regression_results(results):
        sns.set_theme(style="whitegrid")
        # Ensure fallback fonts for English
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(1, 2, figsize=(22, 6))
        fig.canvas.manager.set_window_title('Regression Models Performance')
        
        sns.barplot(x=results['models'], y=results['rmse'], hue=results['models'], palette="Blues_d", legend=False, ax=axes[0])
        axes[0].set_title('Prediction Error (RMSE) by Model Structure', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('RMSE (Lower is better)', fontsize=12)
        axes[0].tick_params(axis='x', labelsize=9)
        axes[0].tick_params(axis='x', rotation=45) 
        for i, v in enumerate(results['rmse']):
            axes[0].text(i, v + (max(results['rmse']) * 0.02), f"{v:.2f}", color='black', ha='center', fontweight='bold')

        sns.barplot(x=results['models'], y=results['accuracy'], hue=results['models'], palette="Oranges_d", legend=False, ax=axes[1])
        axes[1].set_title('Directional Accuracy by Model Structure', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%) (Higher is better)', fontsize=12)
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis='x', labelsize=9)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(50, color='red', linestyle='--', linewidth=2, label='50% Boundary')
        axes[1].legend()

        plt.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.22, wspace=0.20)
        plt.show(block=False)
        plt.pause(1)

    eval_results = {'models': [], 'rmse': [], 'accuracy': []}
    regression_datasets = {
        "Standard Scaling": (standard_train_df, standard_test_df),
        "Robust Scaling": (robust_train_df, robust_test_df),
        "Raw Data": (train_df, test_df)
    }
    
    for scale_name, (tr_df, te_df) in regression_datasets.items():
        print(f"\n{'='*20} [{scale_name}] Dataset Analysis {'='*20}")
        t_df, v_df = tr_df.copy(), te_df.copy()
        t_df['Target_Reg'] = t_df['Close'].shift(-1)
        v_df['Target_Reg'] = v_df['Close'].shift(-1)
        
        t_df = t_df.dropna().reset_index(drop=True)
        v_df = v_df.dropna().reset_index(drop=True)
        
        drop_cols = ['Date', 'Date_only', 'Target', 'Target_Reg'] if 'Target' in t_df.columns else ['Date', 'Date_only', 'Target_Reg']
        X_train_r = t_df.drop(columns=drop_cols, errors='ignore')
        y_train_r = t_df['Target_Reg']
        X_test_r = v_df.drop(columns=drop_cols, errors='ignore')
        y_test_r = v_df['Target_Reg']
        
        models_r = {
            "Linear Regression": LinearRegression(),
            "RF (Base)": RandomForestRegressor(n_estimators=100, random_state=42),
            "RF (max_depth=3)": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }
        
        for model_name, model in models_r.items():
            print(f"\nModel: {model_name}")
            model.fit(X_train_r, y_train_r)
            y_pred = model.predict(X_test_r)
            
            mae = mean_absolute_error(y_test_r, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
            r2 = r2_score(y_test_r, y_pred)
            print(f"  [Overall Error] MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")
            
            bull_indices = v_df[v_df['Bull_Bear'] == 1].index
            bear_indices = v_df[v_df['Bull_Bear'] == 0].index
            
            if len(bull_indices) > 0:
                bull_rmse = np.sqrt(mean_squared_error(y_test_r.iloc[bull_indices], y_pred[bull_indices]))
                print(f"    - Bull Market RMSE: {bull_rmse:.4f}")
            if len(bear_indices) > 0:
                bear_rmse = np.sqrt(mean_squared_error(y_test_r.iloc[bear_indices], y_pred[bear_indices]))
                print(f"    - Bear Market RMSE: {bear_rmse:.4f}")
                
            today_close = v_df['Close']
            y_pred_direction = np.where(y_pred > today_close, 1, 0)
            actual_direction = v_df['Target']
            
            acc = accuracy_score(actual_direction, y_pred_direction)
            print(f"  [Directional Prediction] Accuracy: {acc*100:.2f}%")
            
            model_label = f"{model_name}\n({scale_name[:3]})"
            eval_results['models'].append(model_label)
            eval_results['rmse'].append(rmse)
            eval_results['accuracy'].append(acc * 100)

    print(f"\n-> Regression models evaluation visualization generated.")
    plot_regression_results(eval_results)


    # ==========================================
    # PART 3: Classification Sentiment Comparison
    # ==========================================
    print("\n" + "=" * 60)
    print("PART 3. CLASSIFICATION SENTIMENT COMPARISON")
    print("=" * 60)
    
    def plot_dynamic_results(full_A, full_B, bull_A, bull_B, bear_A, bear_B):
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(labels))
        width = 0.35
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.canvas.manager.set_window_title('Performance Metrics (Ver A vs Ver B)')

        def create_bar(ax, data_A, data_B, title):
            rects1 = ax.bar(x - width / 2, data_A, width, label='Version A (Base)', color='#4C72B0')
            rects2 = ax.bar(x + width / 2, data_B, width, label='Version B (+Sentiment)', color='#C44E52')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            ax.set_ylim(0, 1.0)
            ax.legend(loc='lower right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            for rect in rects1 + rects2:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        create_bar(axes[0], full_A, full_B, 'Full Period Performance')
        create_bar(axes[1], bull_A, bull_B, 'Bull Market (Uptrend)')
        create_bar(axes[2], bear_A, bear_B, 'Bear Market (Downtrend)')
        plt.tight_layout()

    def plot_confusion_matrices(cm_full_A, cm_full_B, cm_bull_A, cm_bull_B, cm_bear_A, cm_bear_B):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.canvas.manager.set_window_title('Confusion Matrices (Ver A vs Ver B)')
        cm_data = [
            (cm_full_A, 'Full Period [Ver A]', axes[0, 0], 'Blues'),
            (cm_bull_A, 'Bull Market [Ver A]', axes[0, 1], 'Blues'),
            (cm_bear_A, 'Bear Market [Ver A]', axes[0, 2], 'Blues'),
            (cm_full_B, 'Full Period [Ver B]', axes[1, 0], 'Reds'),
            (cm_bull_B, 'Bull Market [Ver B]', axes[1, 1], 'Reds'),
            (cm_bear_B, 'Bear Market [Ver B]', axes[1, 2], 'Reds')
        ]
        for cm, title, ax, cmap in cm_data:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap=cmap, values_format='d', colorbar=False)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        plt.tight_layout()

    def plot_roc_pr_curves(y_full, proba_full_A, proba_full_B, y_bull, proba_bull_A, proba_bull_B, y_bear, proba_bear_A, proba_bear_B):
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.canvas.manager.set_window_title('ROC & P-R Curves (Ver A vs Ver B)')
        def draw_roc(ax, y_true, proba_A, proba_B, title):
            if len(y_true) == 0: return
            fpr_A, tpr_A, _ = roc_curve(y_true, proba_A)
            roc_auc_A = auc(fpr_A, tpr_A)
            fpr_B, tpr_B, _ = roc_curve(y_true, proba_B)
            roc_auc_B = auc(fpr_B, tpr_B)
            ax.plot(fpr_A, tpr_A, color='#4C72B0', lw=2, label=f'Ver A (AUC = {roc_auc_A:.3f})')
            ax.plot(fpr_B, tpr_B, color='#C44E52', lw=2, label=f'Ver B (AUC = {roc_auc_B:.3f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (FPR)')
            ax.set_ylabel('True Positive Rate (TPR)')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
        def draw_pr(ax, y_true, proba_A, proba_B, title):
            if len(y_true) == 0: return
            precision_A, recall_A, _ = precision_recall_curve(y_true, proba_A)
            ap_A = average_precision_score(y_true, proba_A)
            precision_B, recall_B, _ = precision_recall_curve(y_true, proba_B)
            ap_B = average_precision_score(y_true, proba_B)
            ax.plot(recall_A, precision_A, color='#4C72B0', lw=2, label=f'Ver A (AP = {ap_A:.3f})')
            ax.plot(recall_B, precision_B, color='#C44E52', lw=2, label=f'Ver B (AP = {ap_B:.3f})')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(loc="lower left")
            ax.grid(alpha=0.3)
        draw_roc(axes[0, 0], y_full, proba_full_A, proba_full_B, 'ROC Curve - Full Period')
        draw_roc(axes[0, 1], y_bull, proba_bull_A, proba_bull_B, 'ROC Curve - Bull Market')
        draw_roc(axes[0, 2], y_bear, proba_bear_A, proba_bear_B, 'ROC Curve - Bear Market')
        draw_pr(axes[1, 0], y_full, proba_full_A, proba_full_B, 'P-R Curve - Full Period')
        draw_pr(axes[1, 1], y_bull, proba_bull_A, proba_bull_B, 'P-R Curve - Bull Market')
        draw_pr(axes[1, 2], y_bear, proba_bear_A, proba_bear_B, 'P-R Curve - Bear Market')
        plt.tight_layout()

    def evaluate_model_cls(df_train, df_test, feature_cols, target_col='Target', n_splits=5, model_name="Model"):
        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        X_test = df_test[feature_cols]
        y_test = df_test[target_col]
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])
        
        param_grid = {
            'classifier__max_depth': [3, 4, 5],
            'classifier__min_samples_split': [5, 10, 15, 20],
            'classifier__min_samples_leaf': [2, 3, 4, 5],
            'classifier__class_weight': [None, 'balanced']
        }
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_cv_acc = grid_search.best_score_
        clean_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        print(f"{model_name} Best Params: {clean_params}")
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        test_acc = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        metrics_list = [test_acc, test_precision, test_recall, test_f1]
        
        return best_cv_acc, metrics_list, cm, y_test, y_pred_proba

    c_train_df = train_df.copy()
    c_test_df = test_df.copy()
    
    base_features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200',
        'EMA_50', 'EMA_200', 'MA_20', 'RSI', 'MACD', 'MACD_Signal',
        'MACD_Hist', 'ADX', 'ATR', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'Daily_Volatility'
    ]
    sentiment_features = [c for c in c_train_df.columns if 'gpt_' in c or 'signed_sentiment_score' in c]
    
    print("\n[ Full Period Model Comparison ]")
    cv_a, metrics_full_A, cm_full_A, y_full, proba_full_A = evaluate_model_cls(
        c_train_df, c_test_df, base_features, model_name="[ Full Period Ver A ]"
    )
    cv_b, metrics_full_B, cm_full_B, _, proba_full_B = evaluate_model_cls(
        c_train_df, c_test_df, base_features + sentiment_features, model_name="[ Full Period Ver B ]"
    )
    
    print(f"[ Full Period A ] Train CV: {cv_a:.4f} | Test Acc: {metrics_full_A[0]:.4f} | Pre: {metrics_full_A[1]:.4f} | Rec: {metrics_full_A[2]:.4f} | F1: {metrics_full_A[3]:.4f}")
    print(f"[ Full Period B ] Train CV: {cv_b:.4f} | Test Acc: {metrics_full_B[0]:.4f} | Pre: {metrics_full_B[1]:.4f} | Rec: {metrics_full_B[2]:.4f} | F1: {metrics_full_B[3]:.4f}")
    
    print("\n[ Bull/Bear Market Comparison ]")
    train_bull, train_bear = c_train_df[c_train_df['Bull_Bear'] == 1], c_train_df[c_train_df['Bull_Bear'] == 0]
    test_bull, test_bear = c_test_df[c_test_df['Bull_Bear'] == 1], c_test_df[c_test_df['Bull_Bear'] == 0]
    
    metrics_bull_A = metrics_bull_B = metrics_bear_A = metrics_bear_B = [0, 0, 0, 0]
    dummy_cm = np.array([[0, 0], [0, 0]])
    cm_bull_A = cm_bull_B = cm_bear_A = cm_bear_B = dummy_cm
    y_bull = y_bear = []
    proba_bull_A = proba_bull_B = proba_bear_A = proba_bear_B = []

    if len(train_bull) > 0 and len(test_bull) > 0:
        bull_cv_a, metrics_bull_A, cm_bull_A, y_bull, proba_bull_A = evaluate_model_cls(
            train_bull, test_bull, base_features, model_name="[ Bull Market Ver A ]"
        )
        bull_cv_b, metrics_bull_B, cm_bull_B, _, proba_bull_B = evaluate_model_cls(
            train_bull, test_bull, base_features + sentiment_features, model_name="[ Bull Market Ver B ]"
        )
        print(f"[ Bull Market A ] Train CV: {bull_cv_a:.4f} | Test Acc: {metrics_bull_A[0]:.4f} | Pre: {metrics_bull_A[1]:.4f} | Rec: {metrics_bull_A[2]:.4f} | F1: {metrics_bull_A[3]:.4f}")
        print(f"[ Bull Market B ] Train CV: {bull_cv_b:.4f} | Test Acc: {metrics_bull_B[0]:.4f} | Pre: {metrics_bull_B[1]:.4f} | Rec: {metrics_bull_B[2]:.4f} | F1: {metrics_bull_B[3]:.4f}")

    if len(train_bear) > 0 and len(test_bear) > 0:
        bear_cv_a, metrics_bear_A, cm_bear_A, y_bear, proba_bear_A = evaluate_model_cls(
            train_bear, test_bear, base_features, model_name="[ Bear Market Ver A ]"
        )
        bear_cv_b, metrics_bear_B, cm_bear_B, _, proba_bear_B = evaluate_model_cls(
            train_bear, test_bear, base_features + sentiment_features, model_name="[ Bear Market Ver B ]"
        )
        print(f"[ Bear Market A ] Train CV: {bear_cv_a:.4f} | Test Acc: {metrics_bear_A[0]:.4f} | Pre: {metrics_bear_A[1]:.4f} | Rec: {metrics_bear_A[2]:.4f} | F1: {metrics_bear_A[3]:.4f}")
        print(f"[ Bear Market B ] Train CV: {bear_cv_b:.4f} | Test Acc: {metrics_bear_B[0]:.4f} | Pre: {metrics_bear_B[1]:.4f} | Rec: {metrics_bear_B[2]:.4f} | F1: {metrics_bear_B[3]:.4f}")

    print(f"-> Classification models evaluation visualization generated (3 windows).")
    plot_dynamic_results(metrics_full_A, metrics_full_B, metrics_bull_A, metrics_bull_B, metrics_bear_A, metrics_bear_B)
    plot_confusion_matrices(cm_full_A, cm_full_B, cm_bull_A, cm_bull_B, cm_bear_A, cm_bear_B)
    plot_roc_pr_curves(y_full, proba_full_A, proba_full_B, y_bull, proba_bull_A, proba_bull_B, y_bear, proba_bear_A, proba_bear_B)
    plt.show(block=False)
    plt.pause(1)


    # ==========================================
    # PART 4: TOP 5 COMBINATION SEARCH 
    # ==========================================
    print("\n" + "=" * 60)
    print("PART 4. TOP 5 COMBINATION SEARCH")
    print("=" * 60)
    
    # 1. Feature sets
    feature_sets = {
        'Base_Features': base_features,
        'Full_Features(Base+Sentiment)': base_features + sentiment_features
    }
    
    # 2. Scalers
    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'Raw(No Scale)': None
    }
    
    # 3. Models
    models = {
        'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {
            'classifier__max_depth': [3, 5],
            'classifier__min_samples_split': [5, 10]
        }),
        'RandomForestClassifier': (RandomForestClassifier(random_state=42), {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5]
        })
    }
    
    top5_results = []
    
    for f_name, f_cols in feature_sets.items():
        X_tr = train_df[f_cols]
        y_tr = train_df['Target']
        X_te = test_df[f_cols]
        y_te = test_df['Target']
        
        for s_name, scaler_obj in scalers.items():
            for m_name, (model_obj, params) in models.items():
                if scaler_obj is not None:
                    pipeline = Pipeline([('scaler', scaler_obj), ('classifier', model_obj)])
                else:
                    pipeline = Pipeline([('classifier', model_obj)])
                
                kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                grid = GridSearchCV(estimator=pipeline, param_grid=params, cv=kfold, scoring='accuracy', n_jobs=-1)
                grid.fit(X_tr, y_tr)
                
                best_mod = grid.best_estimator_
                y_p = best_mod.predict(X_te)
                acc = accuracy_score(y_te, y_p)
                
                top5_results.append({
                    'FeatureSet': f_name,
                    'Scaler': s_name,
                    'Algorithm': m_name,
                    'BestParams': grid.best_params_,
                    'Accuracy': acc
                })
                
    # Sort and print Top 5
    top5_results.sort(key=lambda x: x['Accuracy'], reverse=True)
    
    print("\n[ FINAL EVALUATION: TOP 5 BEST COMBINATIONS BY ACCURACY ]")
    for i, res in enumerate(top5_results[:5]):
        print(f"\n[ Rank {i+1} ]")
        print(f"Algorithm : {res['Algorithm']}")
        print(f"Features  : {res['FeatureSet']}")
        print(f"Scaler    : {res['Scaler']}")
        clean_p = {k.replace('classifier__', ''): v for k, v in res['BestParams'].items()}
        print(f"Params    : {clean_p}")
        print(f"Accuracy  : {res['Accuracy']:.4f}")
        
    print("\nAll integrated analysis pipelines completed.")
    print("Please review the 4 visualization windows that have been opened.")
    print("Close all the plot windows to terminate the program properly.")
    
    plt.show()

if __name__ == "__main__":
    DATA_FILE = "../apple_stock_enriched_phase2_output.csv"
    run_integrated_analysis(DATA_FILE)
