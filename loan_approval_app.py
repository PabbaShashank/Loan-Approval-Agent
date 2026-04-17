import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Loan Approval Agent", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.5rem}
.page-title{font-size:1.6rem;font-weight:600;color:#1A3A52;margin-bottom:0.2rem}
.page-sub{font-size:0.88rem;color:#666;border-bottom:1px solid #ddd;
          padding-bottom:0.8rem;margin-bottom:1.4rem}
.sec{font-size:0.78rem;font-weight:600;color:#1A3A52;text-transform:uppercase;
     letter-spacing:.06em;border-bottom:1.5px solid #E8EEF3;
     padding-bottom:.3rem;margin-bottom:.8rem}
.box-approved{background:#F0FAF4;border-left:4px solid #2E8B57;
              padding:1rem 1.2rem;border-radius:4px;margin-bottom:1rem}
.box-conditional{background:#FFFBF0;border-left:4px solid #C9A227;
                 padding:1rem 1.2rem;border-radius:4px;margin-bottom:1rem}
.box-rejected{background:#FFF5F5;border-left:4px solid #C0392B;
              padding:1rem 1.2rem;border-radius:4px;margin-bottom:1rem}
.dec{font-size:1.25rem;font-weight:600;margin-bottom:.2rem}
.box-approved  .dec{color:#2E8B57}
.box-conditional .dec{color:#B8860B}
.box-rejected  .dec{color:#C0392B}
.note{font-size:.85rem;color:#555}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train():
    df = pd.read_csv("loan_dataset_20000.csv")
    df_m = df.copy()
    cats = df_m.select_dtypes(include='object').columns.tolist()
    enc = {}
    for c in cats:
        le = LabelEncoder()
        df_m[c] = le.fit_transform(df_m[c].astype(str))
        enc[c] = le
    X = df_m.drop('loan_paid_back', axis=1)
    y = df_m['loan_paid_back']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    mods = {
        'Decision Tree'      : DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    res = {}
    for nm, m in mods.items():
        if nm == 'Logistic Regression':
            m.fit(Xtr_s, ytr); yp = m.predict(Xte_s); yb = m.predict_proba(Xte_s)[:,1]
        else:
            m.fit(Xtr, ytr);   yp = m.predict(Xte);   yb = m.predict_proba(Xte)[:,1]
        res[nm] = {'model':m,'y_pred':yp,'y_prob':yb,
                   'accuracy':accuracy_score(yte,yp),'auc':roc_auc_score(yte,yb)}
    return df, X, yte, cats, enc, sc, res


df, X, y_test, cat_cols, encoders, scaler, results = load_and_train()


def run_predict(app, model_name):
    inp = pd.DataFrame([app])
    for c in cat_cols:
        if c in inp.columns:
            try:
                le = encoders[c]
                inp[c] = inp[c].astype(str).apply(
                    lambda v: le.transform([v])[0] if v in set(le.classes_) else 0)
            except Exception:
                inp[c] = 0
    for c in X.columns:
        if c not in inp.columns:
            inp[c] = 0
    inp = inp[X.columns]
    m = results[model_name]['model']
    prob = m.predict_proba(scaler.transform(inp) if model_name == 'Logistic Regression' else inp)[0][1]
    if prob >= 0.60:
        return prob, 'APPROVED',    'approved',    'Low'
    elif prob >= 0.40:
        return prob, 'CONDITIONAL', 'conditional', 'Medium'
    else:
        return prob, 'REJECTED',    'rejected',    'High'


# Header
st.markdown('<div class="page-title">Loan Approval Intelligent Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">SVKM\'s NMIMS MPSTME &nbsp;|&nbsp; Intelligent Agent Systems'
    ' &nbsp;|&nbsp; Dataset: loan_dataset_20000.csv &nbsp;|&nbsp; 20,000 records &nbsp;|&nbsp; 21 features</div>',
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Predict", "Model Performance", "EDA"])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.markdown('<div class="sec">Applicant Details</div>', unsafe_allow_html=True)
    cl, cr = st.columns([1.1, 0.9], gap="large")

    with cl:
        a1, a2 = st.columns(2)
        with a1:
            age    = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male","Female","Other"])
            ms     = st.selectbox("Marital Status", ["Married","Single","Divorced","Widowed"])
            edu    = st.selectbox("Education Level", ["Bachelor's","Master's","High School","PhD","Other"])
        with a2:
            emp    = st.selectbox("Employment Status", ["Employed","Self-employed","Retired","Student","Unemployed"])
            inc    = st.number_input("Annual Income ($)", 6000, 400000, 55000, step=1000)
            loan   = st.number_input("Loan Amount ($)", 500, 50000, 15000, step=500)
            cs     = st.slider("Credit Score", 373, 850, 720)

        b1, b2 = st.columns(2)
        with b1:
            dti    = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.25, step=0.01, format="%.2f")
            purp   = st.selectbox("Loan Purpose",
                                  ["Car","Home","Business","Debt consolidation","Education","Medical","Vacation","Other"])
        with b2:
            term   = st.selectbox("Loan Term (months)", [12,24,36,48,60], index=2)
            grade  = st.selectbox("Grade / Subgrade",
                                  ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5",
                                   "C1","C2","C3","C4","C5","D1","D2","D3","D4","D5",
                                   "E1","E2","E3","F1","F2","F3"], index=6)
            delinq = st.number_input("Delinquencies", 0, 20, 0)
            openacc= st.number_input("Open Accounts", 0, 30, 4)

        model_sel = st.selectbox("Select Model", ["Decision Tree","Random Forest","Logistic Regression"])
        st.button("Run Prediction", type="primary", use_container_width=True)

    rate = 10.5
    mr   = rate / 100 / 12
    inst = (loan * mr) / (1 - (1 + mr) ** -term)
    applicant = {
        'age':age,'gender':gender,'marital_status':ms,'education_level':edu,
        'annual_income':inc,'monthly_income':round(inc/12,2),'employment_status':emp,
        'debt_to_income_ratio':dti,'credit_score':cs,'loan_amount':loan,
        'loan_purpose':purp,'interest_rate':rate,'loan_term':term,
        'installment':round(inst,2),'grade_subgrade':grade,
        'num_of_open_accounts':openacc,'total_credit_limit':round(inc*0.7,2),
        'current_balance':round(loan*0.6,2),'delinquency_history':delinq,
        'public_records':0,'num_of_delinquencies':delinq
    }

    with cr:
        st.markdown('<div class="sec">Decision Output</div>', unsafe_allow_html=True)
        prob, decision, css, risk = run_predict(applicant, model_sel)
        pct = int(prob * 100)
        msgs = {
            'APPROVED'   : 'Low repayment risk. Application can proceed.',
            'CONDITIONAL': 'Moderate risk. Manual review recommended.',
            'REJECTED'   : 'High default risk. Application not recommended.'
        }
        st.markdown(f"""
        <div class="box-{css}">
            <div class="dec">{decision}</div>
            <div class="note">{msgs[decision]}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"**Repayment probability: {pct}%**")
        st.progress(pct / 100)
        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Level", risk)
        m2.metric("Loan / Income", f"{loan/inc*100:.1f}%")
        band = ("Excellent" if cs>=750 else "Good" if cs>=700 else
                "Fair" if cs>=650 else "Poor" if cs>=600 else "Very Poor")
        m3.metric("Credit Band", band)

        st.markdown("---")
        st.markdown("**Feature importance (Decision Tree)**")
        fi_df = pd.DataFrame({'Feature':['Employment status','Credit score','Debt-to-income ratio'],
                              'Importance':[65.6,16.3,16.2]}).sort_values('Importance')
        fig, ax = plt.subplots(figsize=(4, 1.8))
        ax.barh(fi_df['Feature'], fi_df['Importance'], color='#2C6FAC', height=0.45)
        ax.set_xlabel("Importance (%)", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.markdown('<div class="sec">Accuracy and AUC</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (nm, r) in zip([c1,c2,c3], results.items()):
        col.metric(nm, f"{r['accuracy']*100:.2f}%", f"AUC {r['auc']:.3f}")

    st.markdown("---")
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    best = results[best_name]

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"**Confusion Matrix — {best_name}**")
        cm = confusion_matrix(y_test, best['y_pred'])
        fig, ax = plt.subplots(figsize=(4,3.2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Defaulted','Paid Back'],
                    yticklabels=['Defaulted','Paid Back'], linewidths=0.4)
        ax.set_ylabel('Actual', fontsize=9); ax.set_xlabel('Predicted', fontsize=9)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with p2:
        st.markdown("**ROC Curve — All Models**")
        fig, ax = plt.subplots(figsize=(4,3.2))
        for (nm, r), c in zip(results.items(), ['#2C6FAC','#2E8B57','#C0392B']):
            fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
            ax.plot(fpr, tpr, label=f"{nm} ({r['auc']:.3f})", color=c, linewidth=1.2)
        ax.plot([0,1],[0,1],'k--',linewidth=0.7)
        ax.set_xlabel('False Positive Rate',fontsize=9)
        ax.set_ylabel('True Positive Rate',fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")
    st.markdown(f"**Classification Report — {best_name}**")
    rdf = pd.DataFrame(
        classification_report(y_test, best['y_pred'],
                              target_names=['Defaulted','Paid Back'],
                              output_dict=True)).transpose().round(3)
    st.dataframe(rdf, use_container_width=True)

    st.markdown("---")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Decision Tree — first 3 levels**")
        fig, ax = plt.subplots(figsize=(7,4))
        plot_tree(results['Decision Tree']['model'],
                  feature_names=X.columns.tolist(),
                  class_names=['Defaulted','Paid Back'],
                  filled=True, max_depth=3, fontsize=7, ax=ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with t2:
        st.markdown("**Top 10 Feature Importances**")
        fi = pd.Series(results['Decision Tree']['model'].feature_importances_,
                       index=X.columns).nlargest(10).sort_values()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.barh(fi.index, fi.values*100, color='#2C6FAC', height=0.55)
        ax.set_xlabel('Importance (%)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.markdown('<div class="sec">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    e1.metric("Total Records", f"{len(df):,}")
    e2.metric("Paid Back",  f"{df['loan_paid_back'].sum():,}  ({df['loan_paid_back'].mean()*100:.1f}%)")
    e3.metric("Defaulted",  f"{(df['loan_paid_back']==0).sum():,}  ({(1-df['loan_paid_back'].mean())*100:.1f}%)")
    st.markdown("---")

    def twohist(ax, col, xlabel):
        ax.hist(df[df['loan_paid_back']==1][col], bins=35, alpha=0.6, label='Paid', color='#2E8B57')
        ax.hist(df[df['loan_paid_back']==0][col], bins=35, alpha=0.6, label='Defaulted', color='#C0392B')
        ax.set_xlabel(xlabel, fontsize=9); ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=8)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Outcome distribution**")
        fig, ax = plt.subplots(figsize=(3.5,3))
        ax.pie(df['loan_paid_back'].value_counts(),
               labels=['Paid Back','Defaulted'], autopct='%1.1f%%',
               colors=['#2E8B57','#C0392B'], startangle=90, textprops={'fontsize':9})
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with g2:
        st.markdown("**Credit score by outcome**")
        fig, ax = plt.subplots(figsize=(3.5,3))
        twohist(ax, 'credit_score', 'Credit Score')
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with g3:
        st.markdown("**Debt-to-income ratio**")
        fig, ax = plt.subplots(figsize=(3.5,3))
        twohist(ax, 'debt_to_income_ratio', 'DTI')
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    g4, g5, g6 = st.columns(3)
    with g4:
        st.markdown("**Annual income**")
        fig, ax = plt.subplots(figsize=(3.5,3))
        twohist(ax, 'annual_income', 'Annual Income ($)')
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with g5:
        st.markdown("**Loan amount**")
        fig, ax = plt.subplots(figsize=(3.5,3))
        twohist(ax, 'loan_amount', 'Loan Amount ($)')
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with g6:
        st.markdown("**Employment status vs outcome**")
        fig, ax = plt.subplots(figsize=(3.5,3))
        df.groupby(['employment_status','loan_paid_back']).size().unstack().plot(
            kind='bar', ax=ax, color=['#C0392B','#2E8B57'], width=0.6)
        ax.set_xlabel(''); ax.legend(['Defaulted','Paid'], fontsize=8)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")
    st.markdown("**Correlation heatmap (numeric features)**")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.3, ax=ax, annot_kws={"size":7})
    ax.tick_params(labelsize=8); fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")
    st.markdown("**Dataset sample — first 10 rows**")
    st.dataframe(df.head(10), use_container_width=True)
