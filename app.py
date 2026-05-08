import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import io

st.set_page_config(page_title="Data Analysis & Prediction App", layout="wide")
st.title("Data Analysis and Prediction App")

# session state
if "df" not in st.session_state:
    st.session_state.df = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "r2" not in st.session_state:
    st.session_state.r2 = None
if "feature_order" not in st.session_state:
    st.session_state.feature_order = None
if "is_classifier" not in st.session_state:
    st.session_state.is_classifier = True
if "trained_target" not in st.session_state:
    st.session_state.trained_target = None
if "trained_num_cols" not in st.session_state:
    st.session_state.trained_num_cols = []
if "trained_cat_cols" not in st.session_state:
    st.session_state.trained_cat_cols = []


# preprocessing
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the notebook's preprocessing steps."""
    # Drop ID / date columns if present
    drop_cols = [c for c in ["Date", "Product ID", "Store ID"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    # Drop columns with >50% missing
    df = df.loc[:, df.isnull().mean() <= 0.5]
    return df


# upload files
st.markdown("---")
st.subheader("Upload File")
uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.session_state.df = preprocess(raw)
    st.session_state.pipeline = None
    st.session_state.r2 = None
    st.session_state.feature_order = None
    st.session_state.trained_target = None
    st.session_state.trained_num_cols = []
    st.session_state.trained_cat_cols = []
    st.success(f"Dataset loaded: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols")

df = st.session_state.df

if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # target selection
    st.markdown("---")
    st.subheader("Select Target")
    col_left, col_right = st.columns([1, 3])
    with col_left:
        target = st.selectbox("Select Target:", options=num_cols)

    # bar charts
    st.markdown("---")
    if cat_cols:
        selected_cat = st.radio(
            "Categorical variable for first chart:",
            options=cat_cols,
            horizontal=True,
        )

        chart_col1, chart_col2 = st.columns(2)

        # Chart 1 – Average target by categorical variable
        with chart_col1:
            avg_df = df.groupby(selected_cat)[target].mean().reset_index()
            fig1 = px.bar(
                avg_df,
                x=selected_cat,
                y=target,
                title=f"Average {target} by {selected_cat}",
                labels={target: f"{target} (average)"},
                color=selected_cat,
                text_auto=".3f",
                template="simple_white",
            )
            fig1.update_layout(title_x=0.5, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 – Correlation strength of numerical variables with target
        with chart_col2:
            other_num = [c for c in num_cols if c != target]
            if other_num:
                corr_vals = df[other_num + [target]].corr()[target].drop(target).abs()
                corr_df = corr_vals.reset_index()
                corr_df.columns = ["Numerical Variables", "Correlation Strength (Absolute Value)"]
                fig2 = px.bar(
                    corr_df,
                    x="Numerical Variables",
                    y="Correlation Strength (Absolute Value)",
                    title=f"Correlation Strength of Numerical Variables with {target}",
                    text_auto=".2f",
                    template="simple_white",
                    color_discrete_sequence=["#636EFA"],
                )
                fig2.update_layout(title_x=0.5)
                st.plotly_chart(fig2, use_container_width=True)

    # training
    st.markdown("---")
    st.subheader("Train Model")

    all_features = [c for c in df.columns if c != target]
    selected_features = []
    cols_per_row = 6
    feature_rows = [all_features[i : i + cols_per_row] for i in range(0, len(all_features), cols_per_row)]
    for row in feature_rows:
        cb_cols = st.columns(len(row))
        for col, feat in zip(cb_cols, row):
            checked = col.checkbox(feat, value=True, key=f"feat_{feat}")
            if checked:
                selected_features.append(feat)

    train_btn = st.button("Train")

    if train_btn:
        if len(selected_features) == 0:
            st.error("Please select at least one feature.")
        else:
            with st.spinner("Training Decision Tree (with hyperparameter tuning)…"):
                X = df[selected_features]
                y = df[target]

                n_unique = y.nunique()
                is_classifier = (n_unique <= 20) or (n_unique / len(y) < 0.05)
                model_cls = DecisionTreeClassifier if is_classifier else DecisionTreeRegressor

                sel_num = [c for c in selected_features if c in num_cols]
                sel_cat = [c for c in selected_features if c in cat_cols]

                num_transformer = SimpleImputer(strategy="mean")
                cat_transformer = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ])

                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", num_transformer, sel_num),
                        ("cat", cat_transformer, sel_cat),
                    ],
                    remainder="drop",
                )

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model_cls()),
                ])

                param_grid = {
                    "model__max_depth": [None, 10, 20, 30],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 5],
                }

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
                grid_search.fit(X_train, y_train)

                best = grid_search.best_estimator_
                y_pred = best.predict(X_test)

                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, y_pred)

                st.session_state.pipeline = best
                st.session_state.r2 = r2
                st.session_state.is_classifier = is_classifier
                st.session_state.feature_order = selected_features
                st.session_state.trained_target = target
                st.session_state.trained_num_cols = num_cols
                st.session_state.trained_cat_cols = cat_cols

    if st.session_state.r2 is not None:
        st.write(f"The R2 score is: **{st.session_state.r2:.2f}**")

    # prediction
    st.markdown("---")
    st.subheader("Predict")

    if st.session_state.pipeline is not None:
        feat_order = st.session_state.feature_order
        pred_target = st.session_state.trained_target
        pred_num_cols = st.session_state.trained_num_cols
        pred_cat_cols = st.session_state.trained_cat_cols
        placeholder = ", ".join(str(f) for f in feat_order)
        hint = f"Enter values in order: {placeholder}"

        pred_col1, pred_col2, pred_col3 = st.columns([3, 1, 3])
        with pred_col1:
            user_input = st.text_input("Feature values (comma-separated):", placeholder=hint)
        with pred_col2:
            predict_btn = st.button("Predict")
        with pred_col3:
            if predict_btn:
                if not user_input.strip():
                    st.error("Please enter feature values.")
                else:
                    try:
                        raw_vals = [v.strip() for v in user_input.split(",")]
                        if len(raw_vals) != len(feat_order):
                            st.error(
                                f"Expected {len(feat_order)} values, got {len(raw_vals)}. "
                                f"Required order: {placeholder}"
                            )
                        else:
                            row = {}
                            for feat, val in zip(feat_order, raw_vals):
                                if feat in pred_num_cols:
                                    row[feat] = float(val)
                                else:
                                    row[feat] = val
                            input_df = pd.DataFrame([row])
                            prediction = st.session_state.pipeline.predict(input_df)[0]
                            st.success(f"Predicted {pred_target} is: **{prediction}**")
                    except ValueError as e:
                        st.error(f"Invalid input: {e}. Make sure numerical fields contain numbers.")
    else:
        st.info("Train a model first before making predictions.")
