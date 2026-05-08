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
from sklearn.metrics import r2_score

st.set_page_config(page_title="Data Analysis & Prediction App", layout="wide")
st.title("Data Analysis and Prediction App")

# session state
for key, default in {
    "df": None,
    "pipeline": None,
    "r2": None,
    "feature_order": None,
    "trained_target": None,
    "trained_num_cols": [],
    "trained_cat_cols": [],
    "is_classifier": True,
    "prediction_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# preprocessing
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["Date", "Product ID", "Store ID"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.loc[:, df.isnull().mean() <= 0.5]
    return df


# upload files
st.markdown("---")
st.subheader("Upload File")
uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded is not None:
    if st.session_state.df is None:
        raw = pd.read_csv(uploaded)
        st.session_state.df = preprocess(raw)
        st.session_state.pipeline = None
        st.session_state.r2 = None
        st.session_state.feature_order = None
        st.session_state.trained_target = None
        st.session_state.trained_num_cols = []
        st.session_state.trained_cat_cols = []
        st.session_state.prediction_result = None
        st.success(f"Dataset loaded: {st.session_state.df.shape[0]} rows")

if st.session_state.df is not None:
    df = st.session_state.df
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # target selection
    st.markdown("---")
    st.subheader("Select Target")
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

        with chart_col1:
            avg_df = df.groupby(selected_cat)[target].mean().reset_index()
            fig1 = px.bar(
                avg_df, x=selected_cat, y=target,
                title=f"Average {target} by {selected_cat}",
                labels={target: f"{target} (average)"},
                color=selected_cat, text_auto=".3f", template="simple_white",
            )
            fig1.update_layout(title_x=0.5, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            other_num = [c for c in num_cols if c != target]
            if other_num:
                corr_vals = df[other_num + [target]].corr()[target].drop(target).abs()
                corr_df = corr_vals.reset_index()
                corr_df.columns = ["Numerical Variables", "Correlation Strength (Absolute Value)"]
                fig2 = px.bar(
                    corr_df, x="Numerical Variables", y="Correlation Strength (Absolute Value)",
                    title=f"Correlation Strength of Numerical Variables with {target}",
                    text_auto=".2f", template="simple_white",
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
    feature_rows = [all_features[i: i + cols_per_row] for i in range(0, len(all_features), cols_per_row)]
    for feat_row in feature_rows:
        cb_cols = st.columns(len(feat_row))
        for cb_col, feat in zip(cb_cols, feat_row):
            if cb_col.checkbox(feat, value=True, key=f"feat_{feat}"):
                selected_features.append(feat)

    if st.button("Train"):
        if len(selected_features) == 0:
            st.error("Please select at least one feature.")
        else:
            with st.spinner("Training Decision Tree (with hyperparameter tuning)..."):
                X = df[selected_features]
                y = df[target]

                n_unique = y.nunique()
                is_classifier = (n_unique <= 20) or (n_unique / len(y) < 0.05)
                model_cls = DecisionTreeClassifier if is_classifier else DecisionTreeRegressor

                sel_num = [c for c in selected_features if c in num_cols]
                sel_cat = [c for c in selected_features if c in cat_cols]

                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="mean"), sel_num),
                        ("cat", Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]), sel_cat),
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

                grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=1)
                grid_search.fit(X_train, y_train)

                best = grid_search.best_estimator_
                r2 = r2_score(y_test, best.predict(X_test))

                st.session_state.pipeline = best
                st.session_state.r2 = r2
                st.session_state.is_classifier = is_classifier
                st.session_state.feature_order = selected_features
                st.session_state.trained_target = target
                st.session_state.trained_num_cols = num_cols
                st.session_state.trained_cat_cols = cat_cols
                st.session_state.prediction_result = None

    if st.session_state.r2 is not None:
        st.write(f"The R2 score is: **{st.session_state.r2:.2f}**")

# prediction
st.markdown("---")
st.subheader("Predict")

if st.session_state.pipeline is None:
    st.info("Train a model first before making predictions.")
else:
    feat_order = st.session_state.feature_order
    pred_target = st.session_state.trained_target
    pred_num_cols = st.session_state.trained_num_cols
    placeholder = ", ".join(str(f) for f in feat_order)

    user_input = st.text_input(
        "Feature values (comma-separated):",
        placeholder=placeholder,
        key="predict_input",
    )

    if st.button("Predict", key="predict_btn"):
        if not user_input.strip():
            st.session_state.prediction_result = ("error", "Please enter feature values.")
        else:
            raw_vals = [v.strip() for v in user_input.split(",")]
            if len(raw_vals) != len(feat_order):
                st.session_state.prediction_result = (
                    "error",
                    f"Expected {len(feat_order)} values, got {len(raw_vals)}. "
                    f"Required order: {placeholder}",
                )
            else:
                try:
                    row = {}
                    for feat, val in zip(feat_order, raw_vals):
                        row[feat] = float(val) if feat in pred_num_cols else val
                    prediction = st.session_state.pipeline.predict(pd.DataFrame([row]))[0]
                    st.session_state.prediction_result = (
                        "success", f"Predicted {pred_target} is: **{prediction}**"
                    )
                except ValueError as e:
                    st.session_state.prediction_result = (
                        "error", f"Invalid input: {e}. Make sure numerical fields contain numbers."
                    )

    if st.session_state.prediction_result is not None:
        kind, msg = st.session_state.prediction_result
        if kind == "success":
            st.success(msg)
        else:
            st.error(msg)
