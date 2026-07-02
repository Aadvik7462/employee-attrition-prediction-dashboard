import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request, Response, send_file
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "dataset"
app.config["ALLOWED_EXTENSIONS"] = {"csv"}

MODEL_PATH = "model/attrition_model.pkl"
SCALER_PATH = "model/scaler.pkl"
ENCODER_PATH = "model/label_encoders.pkl"
FEATURES_PATH = "model/features.pkl"
DATA_PATH = "dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"

model = None
scaler = None
label_encoders = None
features = None


def load_model_files():
    global model, scaler, label_encoders, features

    if model is None:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoders = joblib.load(ENCODER_PATH)
        features = joblib.load(FEATURES_PATH)

    return model, scaler, label_encoders, features
prediction_history = []
latest_prediction_report = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def train_attrition_model():
    df = pd.read_csv(DATA_PATH)

    df.drop(
        columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"],
        inplace=True,
        errors="ignore"
    )

    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    label_encoders_new = {}

    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders_new[col] = le

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler_new = StandardScaler()
    X_train_scaled = scaler_new.fit_transform(X_train)
    X_test_scaled = scaler_new.transform(X_test)

    model_new = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model_new.fit(X_train_scaled, y_train)

    y_pred = model_new.predict(X_test_scaled)
    y_prob = model_new.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_prob) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model_new.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    os.makedirs("model", exist_ok=True)

    joblib.dump(model_new, MODEL_PATH)
    joblib.dump(scaler_new, SCALER_PATH)
    joblib.dump(label_encoders_new, ENCODER_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)
    joblib.dump(feature_importance, "model/feature_importance.pkl")

    with open("model/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    default_metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "roc_auc": 0,
        "confusion_matrix": [[0, 0], [0, 0]]
    }

    if not os.path.exists(DATA_PATH):
        return render_template(
            "dashboard.html",
            total_employees=0,
            attrition_count=0,
            active_count=0,
            attrition_rate=0,
            avg_age=0,
            avg_income=0,
            avg_years=0,
            avg_promotion=0,
            insights={},
            charts={},
            model_metrics=default_metrics,
            departments=[],
            job_roles=[],
            genders=[],
            attritions=[],
            education_fields=[],
            business_travels=[],
            selected_department="",
            selected_job_role="",
            selected_gender="",
            selected_attrition="",
            selected_education="",
            selected_travel="",
            filtered_records=0
        )

    df = pd.read_csv(DATA_PATH)

    full_df = df.copy()

    selected_department = request.args.get("department", "")
    selected_job_role = request.args.get("job_role", "")
    selected_gender = request.args.get("gender", "")
    selected_attrition = request.args.get("attrition", "")
    selected_education = request.args.get("education_field", "")
    selected_travel = request.args.get("business_travel", "")

    departments = sorted(full_df["Department"].dropna().unique())
    job_roles = sorted(full_df["JobRole"].dropna().unique())
    genders = sorted(full_df["Gender"].dropna().unique())
    attritions = sorted(full_df["Attrition"].dropna().unique())
    education_fields = sorted(full_df["EducationField"].dropna().unique())
    business_travels = sorted(full_df["BusinessTravel"].dropna().unique())

    if selected_department:
        df = df[df["Department"] == selected_department]

    if selected_job_role:
        df = df[df["JobRole"] == selected_job_role]

    if selected_gender:
        df = df[df["Gender"] == selected_gender]

    if selected_attrition:
        df = df[df["Attrition"] == selected_attrition]

    if selected_education:
        df = df[df["EducationField"] == selected_education]

    if selected_travel:
        df = df[df["BusinessTravel"] == selected_travel]

    if df.empty:
        df = full_df.copy()

    total_employees = len(df)
    attrition_count = df[df["Attrition"] == "Yes"].shape[0]
    active_count = df[df["Attrition"] == "No"].shape[0]
    attrition_rate = round((attrition_count / total_employees) * 100, 2) if total_employees > 0 else 0

    avg_age = round(df["Age"].mean(), 1)
    avg_income = round(df["MonthlyIncome"].mean(), 2)
    avg_years = round(df["YearsAtCompany"].mean(), 1)
    avg_promotion = round(df["YearsSinceLastPromotion"].mean(), 1)

    charts = {}

    charts["department"] = px.histogram(df, x="Department", color="Attrition", barmode="group", title="Attrition by Department").to_html(full_html=False)
    charts["jobrole"] = px.histogram(df, y="JobRole", color="Attrition", barmode="group", title="Attrition by Job Role").to_html(full_html=False)
    charts["gender"] = px.histogram(df, x="Gender", color="Attrition", barmode="group", title="Attrition by Gender").to_html(full_html=False)
    charts["overtime"] = px.histogram(df, x="OverTime", color="Attrition", barmode="group", title="Attrition by Overtime").to_html(full_html=False)
    charts["marital"] = px.histogram(df, x="MaritalStatus", color="Attrition", barmode="group", title="Attrition by Marital Status").to_html(full_html=False)

    charts["age"] = px.histogram(df, x="Age", color="Attrition", nbins=20, title="Age Distribution by Attrition").to_html(full_html=False)
    charts["income"] = px.histogram(df, x="MonthlyIncome", color="Attrition", nbins=25, title="Monthly Income Distribution").to_html(full_html=False)
    charts["years"] = px.histogram(df, x="YearsAtCompany", color="Attrition", nbins=20, title="Years at Company Distribution").to_html(full_html=False)

    charts["education_field"] = px.histogram(df, y="EducationField", color="Attrition", barmode="group", title="Attrition by Education Field").to_html(full_html=False)
    charts["business_travel"] = px.histogram(df, x="BusinessTravel", color="Attrition", barmode="group", title="Attrition by Business Travel").to_html(full_html=False)
    charts["job_satisfaction"] = px.histogram(df, x="JobSatisfaction", color="Attrition", barmode="group", title="Attrition by Job Satisfaction").to_html(full_html=False)
    charts["work_life"] = px.histogram(df, x="WorkLifeBalance", color="Attrition", barmode="group", title="Attrition by Work Life Balance").to_html(full_html=False)
    charts["performance"] = px.histogram(df, x="PerformanceRating", color="Attrition", barmode="group", title="Attrition by Performance Rating").to_html(full_html=False)
    charts["environment"] = px.histogram(df, x="EnvironmentSatisfaction", color="Attrition", barmode="group", title="Attrition by Environment Satisfaction").to_html(full_html=False)
    charts["stock"] = px.histogram(df, x="StockOptionLevel", color="Attrition", barmode="group", title="Attrition by Stock Option Level").to_html(full_html=False)

    attrition_yes = df[df["Attrition"] == "Yes"]

    if not attrition_yes.empty:
        highest_dept = attrition_yes["Department"].value_counts().idxmax()
        highest_role = attrition_yes["JobRole"].value_counts().idxmax()
    else:
        highest_dept = "No attrition in current filter"
        highest_role = "No attrition in current filter"

    insights = {
        "highest_dept": highest_dept,
        "highest_role": highest_role,
        "overtime_count": df[df["OverTime"] == "Yes"].shape[0],
        "low_job_satisfaction": df[df["JobSatisfaction"] <= 2].shape[0],
        "highest_income_dept": df.groupby("Department")["MonthlyIncome"].mean().idxmax()
    }

    metrics_path = "model/model_metrics.json"
    importance_path = "model/feature_importance.pkl"

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            model_metrics = json.load(f)
    else:
        model_metrics = default_metrics

    if os.path.exists(importance_path):
        feature_importance = joblib.load(importance_path).head(12)

        charts["feature_importance"] = px.bar(
            feature_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 12 Feature Importance"
        ).to_html(full_html=False)
    else:
        charts["feature_importance"] = ""

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    charts["correlation"] = px.imshow(
        corr,
        title="Correlation Heatmap",
        aspect="auto"
    ).to_html(full_html=False)

    encoded_df = df.copy()

    encoded_df = encoded_df.drop(
        columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"],
        errors="ignore"
    )

    encoded_df["Attrition"] = encoded_df["Attrition"].map({"Yes": 1, "No": 0})

    for col in encoded_df.select_dtypes(include=["object"]).columns:
        if col in label_encoders:
            encoded_df[col] = label_encoders[col].transform(encoded_df[col])

    probability_data = encoded_df[features]
    probability_scaled = scaler.transform(probability_data)
    probability_values = model.predict_proba(probability_scaled)[:, 1] * 100

    probability_df = pd.DataFrame({
        "Attrition Probability": probability_values
    })

    charts["probability_distribution"] = px.histogram(
        probability_df,
        x="Attrition Probability",
        nbins=25,
        title="Prediction Probability Distribution"
    ).to_html(full_html=False)

    return render_template(
        "dashboard.html",
        total_employees=total_employees,
        attrition_count=attrition_count,
        active_count=active_count,
        attrition_rate=attrition_rate,
        avg_age=avg_age,
        avg_income=avg_income,
        avg_years=avg_years,
        avg_promotion=avg_promotion,
        insights=insights,
        charts=charts,
        model_metrics=model_metrics,
        departments=departments,
        job_roles=job_roles,
        genders=genders,
        attritions=attritions,
        education_fields=education_fields,
        business_travels=business_travels,
        selected_department=selected_department,
        selected_job_role=selected_job_role,
        selected_gender=selected_gender,
        selected_attrition=selected_attrition,
        selected_education=selected_education,
        selected_travel=selected_travel,
        filtered_records=len(df)
    )


@app.route("/admin", methods=["GET", "POST"])
def admin():
    global model, scaler, label_encoders, features

    message = None
    error = None
    activity_logs = []

    required_columns = [
        "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
        "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
        "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
        "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
        "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
        "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
        "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    if request.method == "POST":
        action = request.form.get("action")

        if action == "retrain":
            if os.path.exists(DATA_PATH):
                try:
                    metrics = train_attrition_model()

                    model = joblib.load(MODEL_PATH)
                    scaler = joblib.load(SCALER_PATH)
                    label_encoders = joblib.load(ENCODER_PATH)
                    features = joblib.load(FEATURES_PATH)

                    message = f"Model retrained successfully. Accuracy: {metrics['accuracy']}%"
                    activity_logs.append("Model retrained successfully.")
                except Exception as e:
                    error = f"Model retraining failed: {str(e)}"
                    activity_logs.append("Model retraining failed.")
            else:
                error = "Dataset not found. Please upload a dataset first."
                activity_logs.append("Retraining failed because dataset was missing.")

        elif action == "upload":
            if "dataset" not in request.files:
                error = "No file selected."
                activity_logs.append("Upload failed because no file was selected.")
            else:
                file = request.files["dataset"]

                if file.filename == "":
                    error = "Please choose a CSV file."
                    activity_logs.append("Upload failed because file name was empty.")
                elif file and allowed_file(file.filename):
                    try:
                        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

                        filename = secure_filename("WA_Fn-UseC_-HR-Employee-Attrition.csv")
                        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                        file.save(save_path)

                        metrics = train_attrition_model()

                        model = joblib.load(MODEL_PATH)
                        scaler = joblib.load(SCALER_PATH)
                        label_encoders = joblib.load(ENCODER_PATH)
                        features = joblib.load(FEATURES_PATH)

                        message = f"Dataset uploaded and model retrained successfully. Accuracy: {metrics['accuracy']}%"
                        activity_logs.append("Dataset uploaded successfully.")
                        activity_logs.append("Model retrained after upload.")
                    except Exception as e:
                        error = f"Upload successful, but model retraining failed: {str(e)}"
                        activity_logs.append("Upload completed but retraining failed.")
                else:
                    error = "Only CSV files are allowed."
                    activity_logs.append("Upload failed because file type was invalid.")

    dataset_exists = os.path.exists(DATA_PATH)

    if dataset_exists:
        df = pd.read_csv(DATA_PATH)

        total_rows = len(df)
        total_columns = len(df.columns)
        columns = df.columns.tolist()

        departments_count = df["Department"].nunique() if "Department" in df.columns else 0
        job_roles_count = df["JobRole"].nunique() if "JobRole" in df.columns else 0
        attrition_count = df[df["Attrition"] == "Yes"].shape[0] if "Attrition" in df.columns else 0
        attrition_rate = round((attrition_count / total_rows) * 100, 2) if total_rows > 0 else 0

        missing_values = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        missing_columns = [col for col in required_columns if col not in df.columns]

        if len(missing_columns) == 0 and missing_values == 0:
            validation_status = "Dataset Valid"
            validation_class = "valid-status"
        else:
            validation_status = "Dataset Needs Review"
            validation_class = "warning-status"

        dataset_size = round(os.path.getsize(DATA_PATH) / 1024, 2)
        last_updated = datetime.fromtimestamp(os.path.getmtime(DATA_PATH)).strftime("%d %b %Y %I:%M %p")
    else:
        total_rows = 0
        total_columns = 0
        columns = []
        departments_count = 0
        job_roles_count = 0
        attrition_rate = 0
        missing_values = 0
        duplicate_rows = 0
        missing_columns = []
        validation_status = "Dataset Missing"
        validation_class = "danger-status"
        dataset_size = 0
        last_updated = "Not available"

    metrics_path = "model/model_metrics.json"

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            model_metrics = json.load(f)
    else:
        model_metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "roc_auc": 0
        }

    if not activity_logs:
        activity_logs = [
            "Admin page opened.",
            "Dataset status checked.",
            "Model metrics loaded."
        ]

    return render_template(
        "admin.html",
        message=message,
        error=error,
        dataset_exists=dataset_exists,
        total_rows=total_rows,
        total_columns=total_columns,
        columns=columns,
        departments_count=departments_count,
        job_roles_count=job_roles_count,
        attrition_rate=attrition_rate,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        missing_columns=missing_columns,
        validation_status=validation_status,
        validation_class=validation_class,
        dataset_size=dataset_size,
        last_updated=last_updated,
        model_metrics=model_metrics,
        activity_logs=activity_logs
    )

@app.route("/employees")
def employees():
    if not os.path.exists(DATA_PATH):
        return render_template(
            "employees.html",
            employees=[],
            columns=[],
            total_employees=0,
            filtered_count=0,
            departments=[],
            job_roles=[],
            genders=[],
            attritions=[],
            selected_department="",
            selected_job_role="",
            selected_gender="",
            selected_attrition="",
            search_query="",
            page=1,
            total_pages=1,
            avg_age=0,
            avg_income=0
        )

    df = pd.read_csv(DATA_PATH)

    search_query = request.args.get("search", "")
    selected_department = request.args.get("department", "")
    selected_job_role = request.args.get("job_role", "")
    selected_gender = request.args.get("gender", "")
    selected_attrition = request.args.get("attrition", "")
    sort_column = request.args.get("sort", "")
    sort_order = request.args.get("order", "asc")

    filtered_df = df.copy()

    if search_query:
        search_query_lower = search_query.lower()
        filtered_df = filtered_df[
            filtered_df.astype(str).apply(
                lambda row: row.str.lower().str.contains(search_query_lower).any(),
                axis=1
            )
        ]

    if selected_department:
        filtered_df = filtered_df[filtered_df["Department"] == selected_department]

    if selected_job_role:
        filtered_df = filtered_df[filtered_df["JobRole"] == selected_job_role]

    if selected_gender:
        filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]

    if selected_attrition:
        filtered_df = filtered_df[filtered_df["Attrition"] == selected_attrition]

    if sort_column and sort_column in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            by=sort_column,
            ascending=(sort_order == "asc")
        )

    page = int(request.args.get("page", 1))
    per_page = 25

    total_employees = len(df)
    filtered_count = len(filtered_df)
    total_pages = max((filtered_count + per_page - 1) // per_page, 1)

    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    end = start + per_page

    paginated_df = filtered_df.iloc[start:end]

    avg_age = round(filtered_df["Age"].mean(), 1) if filtered_count > 0 else 0
    avg_income = round(filtered_df["MonthlyIncome"].mean(), 2) if filtered_count > 0 else 0

    return render_template(
        "employees.html",
        employees=paginated_df.to_dict(orient="records"),
        columns=paginated_df.columns.tolist(),
        total_employees=total_employees,
        filtered_count=filtered_count,
        departments=sorted(df["Department"].dropna().unique()),
        job_roles=sorted(df["JobRole"].dropna().unique()),
        genders=sorted(df["Gender"].dropna().unique()),
        attritions=sorted(df["Attrition"].dropna().unique()),
        selected_department=selected_department,
        selected_job_role=selected_job_role,
        selected_gender=selected_gender,
        selected_attrition=selected_attrition,
        search_query=search_query,
        page=page,
        total_pages=total_pages,
        sort_column=sort_column,
        sort_order=sort_order,
        avg_age=avg_age,
        avg_income=avg_income
    )

@app.route("/employees/export")
def export_employees():
    if not os.path.exists(DATA_PATH):
        return "Dataset not found"

    df = pd.read_csv(DATA_PATH)

    search_query = request.args.get("search", "")
    selected_department = request.args.get("department", "")
    selected_job_role = request.args.get("job_role", "")
    selected_gender = request.args.get("gender", "")
    selected_attrition = request.args.get("attrition", "")

    filtered_df = df.copy()

    if search_query:
        search_query_lower = search_query.lower()
        filtered_df = filtered_df[
            filtered_df.astype(str).apply(
                lambda row: row.str.lower().str.contains(search_query_lower).any(),
                axis=1
            )
        ]

    if selected_department:
        filtered_df = filtered_df[filtered_df["Department"] == selected_department]

    if selected_job_role:
        filtered_df = filtered_df[filtered_df["JobRole"] == selected_job_role]

    if selected_gender:
        filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]

    if selected_attrition:
        filtered_df = filtered_df[filtered_df["Attrition"] == selected_attrition]

    csv_data = filtered_df.to_csv(index=False)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=filtered_employee_data.csv"
        }
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    global prediction_history, latest_prediction_report

    prediction = None
    probability = None
    result_class = None
    risk_level = None
    risk_color = None
    recommendations = []

    if not os.path.exists(DATA_PATH):
        return render_template(
            "predict.html",
            features=features,
            dropdown_options={},
            numeric_features=[],
            prediction=prediction,
            probability=probability,
            result_class=result_class,
            risk_level=risk_level,
            risk_color=risk_color,
            recommendations=recommendations,
            prediction_history=prediction_history
        )

    df = pd.read_csv(DATA_PATH)

    categorical_features = list(label_encoders.keys())
    numeric_features = [feature for feature in features if feature not in categorical_features]

    dropdown_options = {}

    for feature in categorical_features:
        if feature in df.columns:
            dropdown_options[feature] = sorted(df[feature].dropna().unique())

    if request.method == "POST":
        input_data = {}
        raw_input_data = {}

        for feature in features:
            value = request.form.get(feature)
            raw_input_data[feature] = value

            if feature in label_encoders:
                encoder = label_encoders[feature]
                value = encoder.transform([value])[0]
            else:
                value = float(value)

            input_data[feature] = value

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        prediction = "Employee May Leave" if pred == 1 else "Employee Will Stay"
        probability = round(prob * 100, 2)

        if probability <= 30:
            risk_level = "Low Risk"
            risk_color = "low-risk"
            recommendations = [
                "Maintain current engagement strategy.",
                "Continue regular feedback and recognition.",
                "Provide career growth opportunities."
            ]
        elif probability <= 70:
            risk_level = "Medium Risk"
            risk_color = "medium-risk"
            recommendations = [
                "Monitor employee engagement closely.",
                "Schedule a manager check-in discussion.",
                "Offer training, recognition, or role clarity support."
            ]
        else:
            risk_level = "High Risk"
            risk_color = "high-risk"
            recommendations = [
                "Review workload, overtime, and work-life balance.",
                "Discuss career growth and promotion opportunities.",
                "Create a retention action plan for the employee."
            ]

        result_class = "danger-result" if pred == 1 else "success-result"

        history_item = {
            "date": datetime.now().strftime("%d %b %Y %I:%M %p"),
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level
        }

        prediction_history.insert(0, history_item)
        prediction_history = prediction_history[:10]

        latest_prediction_report = {
            "date": history_item["date"],
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "inputs": raw_input_data
        }

    return render_template(
        "predict.html",
        features=features,
        dropdown_options=dropdown_options,
        numeric_features=numeric_features,
        prediction=prediction,
        probability=probability,
        result_class=result_class,
        risk_level=risk_level,
        risk_color=risk_color,
        recommendations=recommendations,
        prediction_history=prediction_history
    )

@app.route("/reports/pdf")
def download_pdf_report():
    if not os.path.exists(DATA_PATH):
        return "Dataset not found"

    df = pd.read_csv(DATA_PATH)

    total_employees = len(df)
    attrition_count = df[df["Attrition"] == "Yes"].shape[0]
    active_count = df[df["Attrition"] == "No"].shape[0]
    attrition_rate = round((attrition_count / total_employees) * 100, 2)

    avg_age = round(df["Age"].mean(), 1)
    avg_income = round(df["MonthlyIncome"].mean(), 2)
    avg_years = round(df["YearsAtCompany"].mean(), 1)

    highest_dept = df[df["Attrition"] == "Yes"]["Department"].value_counts().idxmax()
    highest_role = df[df["Attrition"] == "Yes"]["JobRole"].value_counts().idxmax()
    overtime_count = df[df["OverTime"] == "Yes"].shape[0]

    metrics_path = "model/model_metrics.json"

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            model_metrics = json.load(f)
    else:
        model_metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "roc_auc": 0
        }

    os.makedirs("reports", exist_ok=True)

    pdf_path = "reports/executive_hr_attrition_report.pdf"

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFillColor(colors.HexColor("#1e3a8a"))
    c.rect(0, height - 90, width, 90, fill=True, stroke=False)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 45, "Employee Attrition Executive Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 65, "Generated from HR Analytics Dashboard")

    y = height - 130

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Dashboard Summary")

    y -= 35

    summary_items = [
        ("Total Employees", total_employees),
        ("Employees Left", attrition_count),
        ("Active Employees", active_count),
        ("Attrition Rate", f"{attrition_rate}%"),
        ("Average Age", avg_age),
        ("Average Monthly Income", f"Rs. {avg_income}"),
        ("Average Years at Company", avg_years)
    ]

    c.setFont("Helvetica", 11)

    for label, value in summary_items:
        c.setFillColor(colors.HexColor("#334155"))
        c.drawString(60, y, f"{label}:")
        c.setFillColor(colors.HexColor("#1e3a8a"))
        c.drawString(250, y, str(value))
        y -= 22

    y -= 20

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Executive Insights")

    y -= 35

    insights = [
        ("Highest Attrition Department", highest_dept),
        ("Highest Attrition Job Role", highest_role),
        ("Employees Working Overtime", overtime_count)
    ]

    c.setFont("Helvetica", 11)

    for label, value in insights:
        c.setFillColor(colors.HexColor("#334155"))
        c.drawString(60, y, f"{label}:")
        c.setFillColor(colors.HexColor("#1e3a8a"))
        c.drawString(260, y, str(value))
        y -= 22

    y -= 20

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Machine Learning Model Performance")

    y -= 35

    ml_items = [
        ("Accuracy", f"{model_metrics.get('accuracy', 0)}%"),
        ("Precision", f"{model_metrics.get('precision', 0)}%"),
        ("Recall", f"{model_metrics.get('recall', 0)}%"),
        ("F1 Score", f"{model_metrics.get('f1_score', 0)}%"),
        ("ROC AUC", f"{model_metrics.get('roc_auc', 0)}%")
    ]

    c.setFont("Helvetica", 11)

    for label, value in ml_items:
        c.setFillColor(colors.HexColor("#334155"))
        c.drawString(60, y, f"{label}:")
        c.setFillColor(colors.HexColor("#1e3a8a"))
        c.drawString(250, y, str(value))
        y -= 22

    y -= 25

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Recommendations")

    y -= 35

    recommendations = [
        "1. Monitor departments and job roles with high attrition.",
        "2. Reduce excessive overtime to improve retention.",
        "3. Improve employee satisfaction and work-life balance.",
        "4. Use ML prediction results to identify high-risk employees early.",
        "5. Review compensation and promotion patterns regularly."
    ]

    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#334155"))

    for rec in recommendations:
        c.drawString(60, y, rec)
        y -= 20

    c.setFillColor(colors.HexColor("#64748b"))
    c.setFont("Helvetica", 9)
    c.drawString(50, 40, "Generated by Employee Attrition Prediction Dashboard")

    c.save()

    return send_file(pdf_path, as_attachment=True)

@app.route("/reports")
def reports():
    dataset_exists = os.path.exists(DATA_PATH)

    if dataset_exists:
        df = pd.read_csv(DATA_PATH)

        total_employees = len(df)
        departments_count = df["Department"].nunique()
        job_roles_count = df["JobRole"].nunique()
        attrition_count = df[df["Attrition"] == "Yes"].shape[0]
        attrition_rate = round((attrition_count / total_employees) * 100, 2)

        metrics_path = "model/model_metrics.json"

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                model_metrics = json.load(f)
            model_accuracy = model_metrics.get("accuracy", 0)
        else:
            model_accuracy = 0
    else:
        total_employees = 0
        departments_count = 0
        job_roles_count = 0
        attrition_rate = 0
        model_accuracy = 0

    return render_template(
        "reports.html",
        total_employees=total_employees,
        departments_count=departments_count,
        job_roles_count=job_roles_count,
        attrition_rate=attrition_rate,
        model_accuracy=model_accuracy
    )


@app.route("/reports/excel")
def download_excel_report():
    if not os.path.exists(DATA_PATH):
        return "Dataset not found"

    df = pd.read_csv(DATA_PATH)

    os.makedirs("reports", exist_ok=True)
    excel_path = "reports/hr_attrition_analytics_report.xlsx"

    total_employees = len(df)
    attrition_count = df[df["Attrition"] == "Yes"].shape[0]
    active_count = df[df["Attrition"] == "No"].shape[0]
    attrition_rate = round((attrition_count / total_employees) * 100, 2)

    summary_df = pd.DataFrame({
        "Metric": [
            "Total Employees",
            "Employees Left",
            "Active Employees",
            "Attrition Rate",
            "Average Age",
            "Average Monthly Income",
            "Average Years at Company"
        ],
        "Value": [
            total_employees,
            attrition_count,
            active_count,
            f"{attrition_rate}%",
            round(df["Age"].mean(), 2),
            round(df["MonthlyIncome"].mean(), 2),
            round(df["YearsAtCompany"].mean(), 2)
        ]
    })

    department_df = df.groupby("Department")["Attrition"].value_counts().unstack(fill_value=0)
    jobrole_df = df.groupby("JobRole")["Attrition"].value_counts().unstack(fill_value=0)
    overtime_df = df.groupby("OverTime")["Attrition"].value_counts().unstack(fill_value=0)

    metrics_path = "model/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        model_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            "Value": [
                metrics.get("accuracy", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1_score", 0),
                metrics.get("roc_auc", 0)
            ]
        })
    else:
        model_df = pd.DataFrame({"Metric": [], "Value": []})

    importance_path = "model/feature_importance.pkl"
    if os.path.exists(importance_path):
        importance_df = joblib.load(importance_path)
    else:
        importance_df = pd.DataFrame({"Feature": [], "Importance": []})

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Dashboard Summary", index=False)
        df.to_excel(writer, sheet_name="Employee Data", index=False)
        department_df.to_excel(writer, sheet_name="Department Analysis")
        jobrole_df.to_excel(writer, sheet_name="Job Role Analysis")
        overtime_df.to_excel(writer, sheet_name="Overtime Analysis")
        model_df.to_excel(writer, sheet_name="Model Performance", index=False)
        importance_df.to_excel(writer, sheet_name="Feature Importance", index=False)

    return send_file(excel_path, as_attachment=True)

@app.route("/reports/ppt")
def download_ppt_report():
    if not os.path.exists(DATA_PATH):
        return "Dataset not found"

    df = pd.read_csv(DATA_PATH)

    total_employees = len(df)
    attrition_count = df[df["Attrition"] == "Yes"].shape[0]
    active_count = df[df["Attrition"] == "No"].shape[0]
    attrition_rate = round((attrition_count / total_employees) * 100, 2)
    avg_age = round(df["Age"].mean(), 1)
    avg_income = round(df["MonthlyIncome"].mean(), 2)

    attrition_yes = df[df["Attrition"] == "Yes"]
    highest_dept = attrition_yes["Department"].value_counts().idxmax()
    highest_role = attrition_yes["JobRole"].value_counts().idxmax()
    overtime_count = df[df["OverTime"] == "Yes"].shape[0]
    low_job_satisfaction = df[df["JobSatisfaction"] <= 2].shape[0]
    highest_income_dept = df.groupby("Department")["MonthlyIncome"].mean().idxmax()

    metrics_path = "model/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            model_metrics = json.load(f)
    else:
        model_metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "roc_auc": 0,
            "confusion_matrix": [[0, 0], [0, 0]]
        }

    importance_path = "model/feature_importance.pkl"
    if os.path.exists(importance_path):
        feature_importance = joblib.load(importance_path).head(10)
    else:
        feature_importance = pd.DataFrame({"Feature": [], "Importance": []})

    os.makedirs("reports", exist_ok=True)

    ppt_path = "reports/employee_attrition_report.pptx"

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    def add_title(slide, title):
        box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.6))
        tf = box.text_frame
        tf.text = title
        p = tf.paragraphs[0]
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = RGBColor(30, 58, 138)

    def add_footer(slide):
        box = slide.shapes.add_textbox(Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3))
        tf = box.text_frame
        tf.text = "Generated by Employee Attrition Prediction Dashboard"
        p = tf.paragraphs[0]
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(100, 116, 139)

    def add_card(slide, x, y, w, h, title, value):
        shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(248, 250, 252)
        shape.line.color.rgb = RGBColor(203, 213, 225)

        tx = slide.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.15), Inches(w - 0.3), Inches(h - 0.3))
        tf = tx.text_frame
        tf.text = str(value)
        p1 = tf.paragraphs[0]
        p1.font.size = Pt(24)
        p1.font.bold = True
        p1.font.color.rgb = RGBColor(30, 64, 175)

        p2 = tf.add_paragraph()
        p2.text = title
        p2.font.size = Pt(12)
        p2.font.color.rgb = RGBColor(51, 65, 85)

    # Slide 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(30, 58, 138)
    bg.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(12), Inches(1.2))
    tf = title_box.text_frame
    tf.text = "Employee Attrition Analytics Report"
    p = tf.paragraphs[0]
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)

    sub_box = slide.shapes.add_textbox(Inches(0.85), Inches(3.5), Inches(11), Inches(0.6))
    tf = sub_box.text_frame
    tf.text = f"Generated on {datetime.now().strftime('%d %b %Y')}"
    p = tf.paragraphs[0]
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(219, 234, 254)

    # Slide 2
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Executive KPI Summary")
    add_card(slide, 0.7, 1.3, 3.0, 1.3, "Total Employees", total_employees)
    add_card(slide, 3.95, 1.3, 3.0, 1.3, "Employees Left", attrition_count)
    add_card(slide, 7.2, 1.3, 3.0, 1.3, "Active Employees", active_count)
    add_card(slide, 0.7, 3.1, 3.0, 1.3, "Attrition Rate", f"{attrition_rate}%")
    add_card(slide, 3.95, 3.1, 3.0, 1.3, "Average Age", avg_age)
    add_card(slide, 7.2, 3.1, 3.0, 1.3, "Avg Monthly Income", f"Rs. {avg_income}")
    add_footer(slide)

    # Slide 3
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Executive Insights")
    add_card(slide, 0.7, 1.3, 3.8, 1.3, "Highest Attrition Department", highest_dept)
    add_card(slide, 4.8, 1.3, 3.8, 1.3, "Highest Attrition Job Role", highest_role)
    add_card(slide, 8.9, 1.3, 3.5, 1.3, "Overtime Employees", overtime_count)
    add_card(slide, 0.7, 3.2, 3.8, 1.3, "Low Job Satisfaction", low_job_satisfaction)
    add_card(slide, 4.8, 3.2, 3.8, 1.3, "Highest Income Department", highest_income_dept)
    add_footer(slide)

    # Slide 4
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Machine Learning Performance")
    add_card(slide, 0.7, 1.3, 2.5, 1.2, "Accuracy", f"{model_metrics.get('accuracy', 0)}%")
    add_card(slide, 3.4, 1.3, 2.5, 1.2, "Precision", f"{model_metrics.get('precision', 0)}%")
    add_card(slide, 6.1, 1.3, 2.5, 1.2, "Recall", f"{model_metrics.get('recall', 0)}%")
    add_card(slide, 8.8, 1.3, 2.5, 1.2, "F1 Score", f"{model_metrics.get('f1_score', 0)}%")
    add_card(slide, 0.7, 3.0, 2.5, 1.2, "ROC AUC", f"{model_metrics.get('roc_auc', 0)}%")

    cm = model_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    add_card(slide, 4.0, 3.0, 2.6, 1.2, "Actual Stay / Pred Stay", cm[0][0])
    add_card(slide, 6.9, 3.0, 2.6, 1.2, "Actual Stay / Pred Leave", cm[0][1])
    add_card(slide, 4.0, 4.7, 2.6, 1.2, "Actual Leave / Pred Stay", cm[1][0])
    add_card(slide, 6.9, 4.7, 2.6, 1.2, "Actual Leave / Pred Leave", cm[1][1])
    add_footer(slide)

    # Slide 5
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Top Feature Importance")

    y = 1.2
    for _, row in feature_importance.iterrows():
        feature = str(row["Feature"])
        importance = round(float(row["Importance"]), 4)

        box = slide.shapes.add_textbox(Inches(0.8), Inches(y), Inches(4.0), Inches(0.3))
        box.text_frame.text = feature
        box.text_frame.paragraphs[0].font.size = Pt(12)

        bar_width = min(6.5, importance * 60)
        bar = slide.shapes.add_shape(1, Inches(4.9), Inches(y), Inches(bar_width), Inches(0.22))
        bar.fill.solid()
        bar.fill.fore_color.rgb = RGBColor(37, 99, 235)
        bar.line.fill.background()

        val = slide.shapes.add_textbox(Inches(11.6), Inches(y - 0.03), Inches(1.0), Inches(0.3))
        val.text_frame.text = str(importance)
        val.text_frame.paragraphs[0].font.size = Pt(11)

        y += 0.45

    add_footer(slide)

    # Slide 6
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Recommendations")

    recommendations = [
        "Monitor departments and job roles with higher attrition risk.",
        "Reduce excessive overtime to improve work-life balance.",
        "Improve employee engagement and satisfaction programs.",
        "Review promotion cycles and career growth opportunities.",
        "Use prediction outputs to identify high-risk employees early.",
        "Review compensation trends across departments and job roles."
    ]

    y = 1.4
    for rec in recommendations:
        box = slide.shapes.add_textbox(Inches(0.9), Inches(y), Inches(11.5), Inches(0.45))
        tf = box.text_frame
        tf.text = "• " + rec
        p = tf.paragraphs[0]
        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor(51, 65, 85)
        y += 0.7

    add_footer(slide)

    prs.save(ppt_path)

    return send_file(ppt_path, as_attachment=True)
@app.route("/reports/snapshot")
def download_dashboard_snapshot():
    if not os.path.exists(DATA_PATH):
        return "Dataset not found"

    df = pd.read_csv(DATA_PATH)

    total_employees = len(df)
    attrition_count = df[df["Attrition"] == "Yes"].shape[0]
    active_count = df[df["Attrition"] == "No"].shape[0]
    attrition_rate = round((attrition_count / total_employees) * 100, 2)
    avg_age = round(df["Age"].mean(), 1)
    avg_income = round(df["MonthlyIncome"].mean(), 2)

    attrition_yes = df[df["Attrition"] == "Yes"]
    highest_dept = attrition_yes["Department"].value_counts().idxmax()
    highest_role = attrition_yes["JobRole"].value_counts().idxmax()

    os.makedirs("reports", exist_ok=True)
    image_path = "reports/dashboard_snapshot.png"

    img = Image.new("RGB", (1400, 900), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 48)
        header_font = ImageFont.truetype("arial.ttf", 30)
        normal_font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 20)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        normal_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.rectangle([0, 0, 1400, 150], fill=(30, 58, 138))
    draw.text((50, 40), "Employee Attrition Dashboard Snapshot", fill="white", font=title_font)
    draw.text((55, 100), "Generated from HR Analytics Platform", fill=(219, 234, 254), font=small_font)

    cards = [
        ("Total Employees", total_employees, (37, 99, 235)),
        ("Employees Left", attrition_count, (239, 68, 68)),
        ("Active Employees", active_count, (16, 185, 129)),
        ("Attrition Rate", f"{attrition_rate}%", (245, 158, 11)),
        ("Average Age", avg_age, (37, 99, 235)),
        ("Avg Monthly Income", f"Rs. {avg_income}", (16, 185, 129)),
    ]

    x_positions = [60, 490, 920]
    y_positions = [220, 430]

    i = 0
    for y in y_positions:
        for x in x_positions:
            title, value, color = cards[i]

            draw.rounded_rectangle([x, y, x + 360, y + 150], radius=20, fill="white", outline=color, width=6)
            draw.text((x + 30, y + 30), str(value), fill=(15, 23, 42), font=header_font)
            draw.text((x + 30, y + 90), title, fill=(51, 65, 85), font=normal_font)

            i += 1

    draw.text((60, 650), "Executive Insights", fill=(15, 23, 42), font=header_font)

    insights = [
        f"Highest Attrition Department: {highest_dept}",
        f"Highest Attrition Job Role: {highest_role}",
        "Recommendation: Monitor overtime, job satisfaction, promotion, and salary trends."
    ]

    y = 710
    for insight in insights:
        draw.text((80, y), "• " + insight, fill=(51, 65, 85), font=normal_font)
        y += 45

    draw.text((60, 850), "Generated by Employee Attrition Prediction Dashboard", fill=(100, 116, 139), font=small_font)

    img.save(image_path)

    return send_file(image_path, as_attachment=True)
@app.route("/predict/report")
def download_prediction_report():
    if not latest_prediction_report:
        return "No prediction report available. Please make a prediction first."

    os.makedirs("reports", exist_ok=True)
    pdf_path = "reports/latest_prediction_report.pdf"

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFillColor(colors.HexColor("#1e3a8a"))
    c.rect(0, height - 90, width, 90, fill=True, stroke=False)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 45, "Employee Attrition Prediction Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 65, f"Generated on {latest_prediction_report['date']}")

    y = height - 130

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Prediction Summary")

    y -= 35

    summary_items = [
        ("Prediction", latest_prediction_report["prediction"]),
        ("Attrition Probability", f"{latest_prediction_report['probability']}%"),
        ("Risk Level", latest_prediction_report["risk_level"])
    ]

    c.setFont("Helvetica", 11)

    for label, value in summary_items:
        c.setFillColor(colors.HexColor("#334155"))
        c.drawString(60, y, f"{label}:")
        c.setFillColor(colors.HexColor("#1e3a8a"))
        c.drawString(230, y, str(value))
        y -= 24

    y -= 20

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Recommendations")

    y -= 35

    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#334155"))

    for rec in latest_prediction_report["recommendations"]:
        c.drawString(70, y, "• " + rec)
        y -= 22

    y -= 20

    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Employee Input Summary")

    y -= 30

    c.setFont("Helvetica", 8)

    count = 0
    for key, value in latest_prediction_report["inputs"].items():
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 8)

        c.setFillColor(colors.HexColor("#334155"))
        c.drawString(60, y, f"{key}: {value}")
        y -= 14
        count += 1

        if count >= 40:
            break

    c.save()

    return send_file(pdf_path, as_attachment=True)
@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html"), 500

if __name__ == "__main__":
    app.run(debug=True)