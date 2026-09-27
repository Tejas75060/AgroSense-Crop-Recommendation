#!/usr/bin/env python3
"""
AgroSense Academic Project Report PDF Generator
Built using ReportLab for Group 15 - Machine Learning Fundamentals Mini Project
Generates a formal, 16-section technical report PDF with cover page, tables, and figures.
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "Project_Report.pdf")
EDA_DIR = os.path.join(BASE_DIR, "eda_plots")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        if self._pageNumber == 1:
            # Skip header and footer on cover page
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4a5568"))

        # Running Header
        self.drawString(54, 800, "Group 15 | AgroSense: Precision Crop Recommendation System")
        self.drawRightString(540, 800, "ML Fundamentals Mini Project")
        self.setStrokeColor(colors.HexColor("#cbd5e0"))
        self.setLineWidth(0.6)
        self.line(54, 794, 540, 794)

        # Running Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.line(54, 48, 540, 48)
        self.drawString(54, 36, "Confidential - Department of Computer Science & Engineering")
        self.drawRightString(540, 36, page_str)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1b4332")
    secondary_color = colors.HexColor("#2d6a4f")
    dark_neutral = colors.HexColor("#1a202c")
    body_color = colors.HexColor("#2d3748")

    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=primary_color,
        alignment=1, # Center
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        alignment=1,
        spaceAfter=25
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=body_color,
        spaceAfter=7
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=body_color,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=body_color
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#718096"),
        alignment=1,
        spaceAfter=8
    )

    story = []

    # =========================================================================
    # 1. COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("MACHINE LEARNING FUNDAMENTALS MINI PROJECT", ParagraphStyle(
        'UpperTag', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10,
        leading=13, textColor=secondary_color, alignment=1, spaceAfter=15
    )))

    story.append(Paragraph("AgroSense: An Intelligent Soil & Climate-Driven Precision Crop Recommendation System", title_style))
    story.append(Paragraph("A Multi-Class Supervised Classification Framework for Precision Agriculture and Yield Optimization", subtitle_style))
    story.append(HRFlowable(width="80%", thickness=1.5, color=secondary_color, spaceBefore=5, spaceAfter=30))

    # Meta Table on Cover
    cover_meta = [
        [Paragraph("<b>Academic Term:</b>", table_cell_style), Paragraph("Fall Semester 2026", table_cell_style)],
        [Paragraph("<b>Course:</b>", table_cell_style), Paragraph("Machine Learning Fundamentals (CS401 / ML202)", table_cell_style)],
        [Paragraph("<b>Group Identifier:</b>", table_cell_style), Paragraph("<b>GROUP 15</b>", table_cell_style)],
        [Paragraph("<b>Assigned Problem:</b>", table_cell_style), Paragraph("Crop Recommendation System (Page 16)", table_cell_style)],
        [Paragraph("<b>Evaluation Max Marks:</b>", table_cell_style), Paragraph("100 Marks (Rubric-Aligned)", table_cell_style)],
        [Paragraph("<b>Primary Algorithms:</b>", table_cell_style), Paragraph("Random Forest, Decision Tree, Naive Bayes, SVM, KNN", table_cell_style)],
        [Paragraph("<b>Deployment:</b>", table_cell_style), Paragraph("Streamlit Cloud / Local Interactive Web Dashboard", table_cell_style)],
        [Paragraph("<b>Date of Submission:</b>", table_cell_style), Paragraph("September 2026", table_cell_style)]
    ]
    t_cover = Table(cover_meta, colWidths=[140, 260])
    t_cover.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#edf2f7")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_cover)

    story.append(Spacer(1, 40))
    story.append(Paragraph("<b>Submitted By:</b>", ParagraphStyle('SubHeading', fontName='Helvetica-Bold', fontSize=10, textColor=primary_color, alignment=1)))
    story.append(Paragraph("Group 15 Project Team Members", ParagraphStyle('SubText', fontName='Helvetica', fontSize=9.5, textColor=body_color, alignment=1, spaceAfter=40)))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceBefore=10, spaceAfter=15))
    story.append(Paragraph("Department of Computer Science and Engineering • Faculty of Technology", ParagraphStyle('Dept', fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor("#718096"), alignment=1)))

    story.append(PageBreak())

    # =========================================================================
    # 2. ABSTRACT
    # =========================================================================
    story.append(Paragraph("Abstract", h1_style))
    story.append(Paragraph(
        "Agriculture remains the socioeconomic cornerstone of emerging economies, yet conventional agricultural decision-making relies heavily on traditional intuition and generalized regional calendars, often resulting in severe crop failure, soil nutrient depletion, and sub-optimal economic yield. "
        "In this project, <b>Group 15</b> presents <b>AgroSense</b>, an intelligent precision crop recommendation system that formulates crop selection as a multi-class supervised learning task based on seven critical soil and environmental parameters: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature (°C), Relative Humidity (%), Soil pH, and Annual Rainfall (mm). "
        "Using a rigorously curated dataset of 2,200 agricultural observations spanning 22 distinct crop categories (100 samples per class), we implemented, benchmarked, and evaluated five classification architectures: Decision Tree, Random Forest Classifier, Gaussian Naive Bayes, Support Vector Machine (RBF kernel), and K-Nearest Neighbors. "
        "The champion model—<b>Random Forest Classifier (100 estimators)</b>—achieved a benchmark test accuracy of <b>99.55%</b>, weighted precision of <b>99.57%</b>, weighted recall of <b>99.55%</b>, weighted F1-score of <b>99.55%</b>, and a 5-fold stratified cross-validation mean of <b>99.55% (± 0.32%)</b>. "
        "Feature importance analysis using Gini impurity reduction revealed that annual rainfall (23.0%) and relative humidity (22.4%) are the primary agro-ecological discriminators, followed by potassium (17.5%) and phosphorus (15.1%). "
        "Finally, the trained ensemble model is encapsulated and deployed as an intuitive, high-performance Streamlit web dashboard supporting single-sample soil diagnostics with confidence intervals, batch CSV file inference, interactive EDA analytics, and an integrated soil health fertilizer advisory system.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 3. INTRODUCTION & MOTIVATION
    # =========================================================================
    story.append(Paragraph("1. Introduction and Motivation", h1_style))
    story.append(Paragraph(
        "Modern agricultural ecosystems face immense challenges stemming from climate change, soil nutrient degradation, erratic rainfall patterns, and depleting water tables. "
        "In developing agrarian nations, a significant proportion of smallholder farmers select crops based on historical habit, ancestral practices, or fleeting market rumors, rather than verifiable laboratory soil chemistry and meteorological conditions. "
        "When an unsuitable crop is planted in an incompatible agro-climatic zone, farmers experience stunted vegetative growth, vulnerability to pests, catastrophic crop failure, and deep debt cycles.",
        body_style
    ))
    story.append(Paragraph(
        "The motivation behind <b>AgroSense</b> is to leverage Machine Learning to democratize precision agriculture. By quantifying the non-linear interactions between soil macronutrients (NPK) and ambient weather variables, our system provides data-driven, localized, and actionable recommendations. "
        "Such precision systems minimize artificial fertilizer wastage, reduce groundwater over-extraction, and empower farmers with predictive foresight before sowing.",
        body_style
    ))

    # =========================================================================
    # 4. PROBLEM STATEMENT & OBJECTIVES
    # =========================================================================
    story.append(Paragraph("2. Problem Statement and Objectives", h1_style))
    story.append(Paragraph("<b>Problem Statement:</b>", h2_style))
    story.append(Paragraph(
        "Given seven numerical soil and environmental attributes—Nitrogen ratio (N), Phosphorus ratio (P), Potassium ratio (K), Temperature (°C), Relative Humidity (%), Soil pH (0–14), and Annual Rainfall (mm)—construct, evaluate, and deploy a robust multi-class Machine Learning classification model that accurately predicts the optimal agricultural crop to maximize productivity and eliminate trial-and-error farming risks.",
        body_style
    ))
    story.append(Paragraph("<b>Project Objectives:</b>", h2_style))
    story.append(Paragraph("• <b>Data Engineering & Quality Assurance:</b> Audit, clean, and verify the integrity of the 2,200-sample precision agriculture dataset ensuring zero missing values or duplicate records.", bullet_style))
    story.append(Paragraph("• <b>Exploratory Data Analysis:</b> Generate publication-grade statistical visualizations to characterize the ecological distributions, feature correlations, and distinct nutrient demands across 22 crops.", bullet_style))
    story.append(Paragraph("• <b>Model Development:</b> Implement Decision Tree and Random Forest classifiers alongside Gaussian Naive Bayes, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN).", bullet_style))
    story.append(Paragraph("• <b>Comprehensive Evaluation:</b> Quantitatively assess all algorithms across Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and 5-Fold Stratified Cross-Validation.", bullet_style))
    story.append(Paragraph("• <b>Interpretability & Feature Importance:</b> Determine the relative importance of environmental vs. soil parameters using Mean Decrease in Impurity (Gini Importance).", bullet_style))
    story.append(Paragraph("• <b>Interactive Web Deployment:</b> Build a responsive, accessible Streamlit web dashboard with real-time parameter validation, confidence scores, batch processing, and agronomic guidance.", bullet_style))

    # =========================================================================
    # 5. LITERATURE SURVEY & EXISTING SYSTEMS
    # =========================================================================
    story.append(Paragraph("3. Literature Survey and Limitations of Existing Systems", h1_style))
    story.append(Paragraph(
        "In recent literature, multiple methodologies have been proposed for smart farming and crop recommendation. "
        "Bondre and Mahagaonkar (2019) demonstrated the efficacy of multi-class classification using Decision Trees and Random Forests, noting that ensemble methods significantly outperform single tree estimators due to variance reduction. "
        "Pudumalar et al. (2016) explored Naive Bayes, KNN, and SVM for precision farming, concluding that while Naive Bayes performs well on small datasets due to the conditional independence assumption, tree ensembles offer greater robustness against outliers and collinear features. "
        "Kumar et al. (2020) implemented deep learning multilayer perceptrons (MLP) for agro-climatic prediction but observed that for tabular agronomic data, tree-based gradient boosting and bagging ensembles achieve equal or superior accuracy with significantly lower computational latency and zero hyper-parameter sensitivity.",
        body_style
    ))
    story.append(Paragraph("<b>Limitations of Existing Approaches:</b>", h2_style))
    story.append(Paragraph("1. <b>Binary / Limited Crop Choices:</b> Most existing tools focus narrowly on 2 to 4 major regional staples (e.g., Rice, Wheat, Cotton), ignoring commercial fruits and pulses.", bullet_style))
    story.append(Paragraph("2. <b>Static Soil Health Cards:</b> Traditional government soil health cards provide physical printouts of NPK numbers without providing predictive crop suitability models.", bullet_style))
    story.append(Paragraph("3. <b>Absence of Uncertainty Quantification:</b> Standard recommendation tools output a single hard label without confidence metrics or second/third-best alternatives.", bullet_style))
    story.append(Paragraph("4. <b>Lack of Batch Diagnostic Tooling:</b> Existing web demos process only one farmer query at a time, rendering them impractical for agricultural extension officers surveying whole villages.", bullet_style))

    # =========================================================================
    # 6. PROPOSED SYSTEM ARCHITECTURE & WORKFLOW
    # =========================================================================
    story.append(Paragraph("4. Proposed AgroSense System Architecture and Workflow", h1_style))
    story.append(Paragraph(
        "The proposed AgroSense system adopts a modular, 10-stage Machine Learning lifecycle designed for reproducibility, statistical validity, and real-time deployment:",
        body_style
    ))

    workflow_data = [
        [Paragraph("<b>Stage</b>", table_header_style), Paragraph("<b>Component</b>", table_header_style), Paragraph("<b>Key Responsibilities & Deliverables</b>", table_header_style)],
        [Paragraph("1", table_cell_style), Paragraph("Problem Formulation", table_cell_style), Paragraph("Formalize 7-feature multi-class classification over 22 crop labels.", table_cell_style)],
        [Paragraph("2", table_cell_style), Paragraph("Data Acquisition", table_cell_style), Paragraph("Load 2,200 records from agricultural research repositories.", table_cell_style)],
        [Paragraph("3", table_cell_style), Paragraph("Data Quality Audit", table_cell_style), Paragraph("Verify 0 null values, 0 duplicate rows, uniform 100-sample class balance.", table_cell_style)],
        [Paragraph("4", table_cell_style), Paragraph("Exploratory Data Analysis", table_cell_style), Paragraph("Generate 6 high-res statistical plots (histograms, heatmap, boxplots, NPK demands).", table_cell_style)],
        [Paragraph("5", table_cell_style), Paragraph("Data Splitting", table_cell_style), Paragraph("Stratified 80:20 split (1,760 train samples, 440 test samples, 20 per crop).", table_cell_style)],
        [Paragraph("6", table_cell_style), Paragraph("Model Training", table_cell_style), Paragraph("Train Random Forest, Decision Tree, Gaussian NB, SVM (RBF), and KNN.", table_cell_style)],
        [Paragraph("7", table_cell_style), Paragraph("Validation & Tuning", table_cell_style), Paragraph("5-Fold Stratified Cross-Validation on all candidate models.", table_cell_style)],
        [Paragraph("8", table_cell_style), Paragraph("Benchmarking & Metrics", table_cell_style), Paragraph("Calculate Accuracy, Precision, Recall, F1, and Confusion Matrices.", table_cell_style)],
        [Paragraph("9", table_cell_style), Paragraph("Model Serialization", table_cell_style), Paragraph("Export trained ensemble pipeline and metadata via joblib serialization.", table_cell_style)],
        [Paragraph("10", table_cell_style), Paragraph("Streamlit Deployment", table_cell_style), Paragraph("Interactive UI with single prediction, confidence scores, and bulk CSV mode.", table_cell_style)]
    ]
    t_workflow = Table(workflow_data, colWidths=[35, 125, 325])
    t_workflow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_workflow)
    story.append(Spacer(1, 10))

    story.append(PageBreak())

    # =========================================================================
    # 7. DATASET DESCRIPTION
    # =========================================================================
    story.append(Paragraph("5. Dataset Description and Statistical Summary", h1_style))
    story.append(Paragraph(
        "The underlying dataset (`Crop_recommendation.csv`) comprises 2,200 instances curated from precision agricultural soil tests and meteorological stations in India. "
        "The dataset features zero missing values, zero duplicates, and a perfectly balanced class distribution with exactly 100 instances for each of the 22 crops. "
        "The crops span four agricultural classifications: Cereals (Rice, Maize), Pulses (Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil), Fruits (Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut), and Cash/Fiber Crops (Cotton, Jute, Coffee).",
        body_style
    ))

    # Feature Summary Table
    feat_data = [
        [Paragraph("<b>Feature</b>", table_header_style), Paragraph("<b>Unit</b>", table_header_style), Paragraph("<b>Min</b>", table_header_style), Paragraph("<b>Mean</b>", table_header_style), Paragraph("<b>Max</b>", table_header_style), Paragraph("<b>Std Dev</b>", table_header_style), Paragraph("<b>Agronomic Role</b>", table_header_style)],
        [Paragraph("Nitrogen (N)", table_cell_style), Paragraph("mg/kg", table_cell_style), Paragraph("0.0", table_cell_style), Paragraph("50.55", table_cell_style), Paragraph("140.0", table_cell_style), Paragraph("36.92", table_cell_style), Paragraph("Leaf growth and chlorophyll synthesis", table_cell_style)],
        [Paragraph("Phosphorus (P)", table_cell_style), Paragraph("mg/kg", table_cell_style), Paragraph("5.0", table_cell_style), Paragraph("53.36", table_cell_style), Paragraph("145.0", table_cell_style), Paragraph("32.99", table_cell_style), Paragraph("Root elongation, blooming, seed formation", table_cell_style)],
        [Paragraph("Potassium (K)", table_cell_style), Paragraph("mg/kg", table_cell_style), Paragraph("5.0", table_cell_style), Paragraph("48.15", table_cell_style), Paragraph("205.0", table_cell_style), Paragraph("50.65", table_cell_style), Paragraph("Osmoregulation, drought and pest resistance", table_cell_style)],
        [Paragraph("Temperature", table_cell_style), Paragraph("°C", table_cell_style), Paragraph("8.83", table_cell_style), Paragraph("25.62", table_cell_style), Paragraph("43.68", table_cell_style), Paragraph("5.06", table_cell_style), Paragraph("Metabolic rate and photosynthetic efficiency", table_cell_style)],
        [Paragraph("Humidity", table_cell_style), Paragraph("%", table_cell_style), Paragraph("14.26", table_cell_style), Paragraph("71.48", table_cell_style), Paragraph("99.98", table_cell_style), Paragraph("22.26", table_cell_style), Paragraph("Transpiration and atmospheric vapor pressure", table_cell_style)],
        [Paragraph("Soil pH", table_cell_style), Paragraph("Scale", table_cell_style), Paragraph("3.50", table_cell_style), Paragraph("6.47", table_cell_style), Paragraph("9.94", table_cell_style), Paragraph("0.77", table_cell_style), Paragraph("Soil chemical acidity / micronutrient solubility", table_cell_style)],
        [Paragraph("Rainfall", table_cell_style), Paragraph("mm", table_cell_style), Paragraph("20.21", table_cell_style), Paragraph("103.46", table_cell_style), Paragraph("298.56", table_cell_style), Paragraph("54.96", table_cell_style), Paragraph("Annual moisture replenishment for root zone", table_cell_style)]
    ]
    t_feat = Table(feat_data, colWidths=[70, 40, 35, 45, 40, 50, 205])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_feat)
    story.append(Paragraph("Table 1: Descriptive statistics and agronomic role of input features.", caption_style))
    story.append(Spacer(1, 10))

    # =========================================================================
    # 8. DATA PREPROCESSING
    # =========================================================================
    story.append(Paragraph("6. Data Preprocessing & Validation", h1_style))
    story.append(Paragraph(
        "A rigorous preprocessing pipeline was executed to guarantee data cleanliness and prevent methodological leaks: "
        "<br/>1. <b>Missing Value Auditing:</b> Verified using `df.isnull().sum()`. Exactly 0 null values exist across all 8 columns. "
        "<br/>2. <b>Deduplication:</b> Verified using `df.duplicated().sum()`. Exactly 0 duplicate rows were detected. "
        "<br/>3. <b>Outlier Treatment Assessment:</b> Boxplot analysis identified apparent outliers in Potassium (K > 150 mg/kg) and Rainfall (Rainfall > 250 mm). However, biological domain verification indicates that these values represent genuine physiological adaptations—for example, Grapes and Apples naturally require extreme Potassium reserves, while Rice and Jute flourish in flooded fields exceeding 200 mm rainfall. Removing these data points would severely impair the model's ability to recommend these specific crops. Consequently, no destructive outlier truncation was performed. "
        "<br/>4. <b>Feature Scaling Analysis:</b> Tree-based algorithms (Decision Tree and Random Forest) operate via greedy orthogonal splits ($x_j \le \theta$) on single features and are inherently scale-invariant. For distance-based estimators (SVM, KNN), a `StandardScaler()` step was incorporated via a scikit-learn `Pipeline` to standardize features to zero mean and unit variance ($\mu=0, \sigma=1$).",
        body_style
    ))

    # =========================================================================
    # 9. EXPLORATORY DATA ANALYSIS (EDA)
    # =========================================================================
    story.append(Paragraph("7. Exploratory Data Analysis (EDA)", h1_style))
    story.append(Paragraph(
        "To comprehend the agro-climatic boundaries separating crops, we generated multiple analytical visualizations:",
        body_style
    ))

    # Insert EDA Images
    eda_corr_img = os.path.join(EDA_DIR, "03_correlation_heatmap.png")
    if os.path.exists(eda_corr_img):
        story.append(Image(eda_corr_img, width=5.5*inch, height=3.6*inch))
        story.append(Paragraph("Figure 1: Pearson Correlation Heatmap across 7 Soil & Meteorological Attributes.", caption_style))

    story.append(Paragraph(
        "<b>Correlation Observations:</b> As illustrated in Figure 1, Phosphorus (P) and Potassium (K) exhibit a substantial positive linear correlation of <b>r = +0.74</b>. This aligns with agronomic reality, as fruits like Grapes and Apples demand concurrent surges of both macronutrients during flower and fruit development. Conversely, environmental factors such as Temperature, Humidity, and Rainfall demonstrate near-zero mutual linear correlation, confirming that they represent independent ecological dimensions.",
        body_style
    ))

    story.append(PageBreak())

    eda_npk_img = os.path.join(EDA_DIR, "05_npk_ratio_by_crop.png")
    if os.path.exists(eda_npk_img):
        story.append(Image(eda_npk_img, width=6.5*inch, height=2.6*inch))
        story.append(Paragraph("Figure 2: Average N-P-K Nutrient Requirements Across 22 Crop Classes.", caption_style))

    story.append(Paragraph(
        "<b>Nutrient Demand Profiles:</b> Figure 2 underscores dramatic class separation based on soil nutrients. "
        "Cotton, Coffee, and Maize exhibit extremely high Nitrogen demands (N > 80 mg/kg), whereas leguminous pulses (Chickpea, Lentil, Mothbeans) require minimal Nitrogen (N < 40 mg/kg) because their symbiotic <i>Rhizobium</i> bacteria fix atmospheric nitrogen. "
        "Grapes and Apples demand unprecedented Potassium (> 190 mg/kg), establishing immediate separability in tree-based decision nodes.",
        body_style
    ))

    # =========================================================================
    # 10. MODEL DEVELOPMENT
    # =========================================================================
    story.append(Paragraph("8. Machine Learning Model Development", h1_style))
    story.append(Paragraph(
        "In accordance with project guidelines, five distinct supervised classification algorithms were implemented and rigorously benchmarked:",
        body_style
    ))
    story.append(Paragraph("<b>1. Decision Tree Classifier:</b> A non-parametric greedy recursive partitioning algorithm utilizing the Gini Impurity metric: "
                           "<br/><i>Gini(D) = 1 - Σ (p_i)^2</i>. While transparent and human-interpretable, individual decision trees are susceptible to high variance and sensitive to training perturbations.", body_style))
    story.append(Paragraph("<b>2. Random Forest Classifier:</b> An ensemble bagging meta-estimator constructing 100 de-correlated decision trees trained on bootstrap sub-samples with random feature subsets (max_features = sqrt). Predictions are aggregated via majority voting, effectively eliminating individual tree variance.", body_style))
    story.append(Paragraph("<b>3. Gaussian Naive Bayes:</b> A probabilistic classifier founded on Bayes' Theorem under the assumption of conditional feature independence given the class label: "
                           "<br/><i>P(y | x1..xn) ∝ P(y) · Π P(xi | y)</i>. Assumes normal distribution of continuous features.", body_style))
    story.append(Paragraph("<b>4. Support Vector Machine (SVM):</b> A maximum-margin hyper-plane classifier utilizing the Radial Basis Function (RBF) non-linear kernel preceded by standard feature scaling.", body_style))
    story.append(Paragraph("<b>5. K-Nearest Neighbors (KNN):</b> An instance-based non-parametric classifier (k=5) determining crop class through Euclidean distance in standardized feature space.", body_style))

    # =========================================================================
    # 11. MODEL EVALUATION & COMPARISON
    # =========================================================================
    story.append(Paragraph("9. Model Evaluation and Quantitative Comparison", h1_style))
    story.append(Paragraph(
        "The models were evaluated on the 440-sample stratified test set (20 samples per crop) and further validated using 5-Fold Stratified Cross-Validation across all 2,200 records. Table 2 summarizes the quantitative findings:",
        body_style
    ))

    # Model Comparison Table
    comp_data = [
        [Paragraph("<b>Model Architecture</b>", table_header_style), Paragraph("<b>Test Accuracy</b>", table_header_style), Paragraph("<b>Precision (Wt)</b>", table_header_style), Paragraph("<b>Recall (Wt)</b>", table_header_style), Paragraph("<b>F1-Score (Wt)</b>", table_header_style), Paragraph("<b>5-Fold CV Mean (± Std)</b>", table_header_style)],
        [Paragraph("<b>Random Forest (Ensemble)</b>", table_cell_style), Paragraph("<b>99.55%</b>", table_cell_style), Paragraph("<b>99.57%</b>", table_cell_style), Paragraph("<b>99.55%</b>", table_cell_style), Paragraph("<b>99.55%</b>", table_cell_style), Paragraph("<b>99.55% (± 0.32%)</b>", table_cell_style)],
        [Paragraph("Gaussian Naive Bayes", table_cell_style), Paragraph("99.55%", table_cell_style), Paragraph("99.59%", table_cell_style), Paragraph("99.55%", table_cell_style), Paragraph("99.54%", table_cell_style), Paragraph("99.45% (± 0.18%)", table_cell_style)],
        [Paragraph("Support Vector Machine (RBF)", table_cell_style), Paragraph("98.86%", table_cell_style), Paragraph("98.96%", table_cell_style), Paragraph("98.86%", table_cell_style), Paragraph("98.87%", table_cell_style), Paragraph("98.82% (± 0.36%)", table_cell_style)],
        [Paragraph("Decision Tree (Single)", table_cell_style), Paragraph("97.95%", table_cell_style), Paragraph("98.06%", table_cell_style), Paragraph("97.95%", table_cell_style), Paragraph("97.94%", table_cell_style), Paragraph("98.77% (± 0.68%)", table_cell_style)],
        [Paragraph("K-Nearest Neighbors (k=5)", table_cell_style), Paragraph("97.95%", table_cell_style), Paragraph("98.04%", table_cell_style), Paragraph("97.95%", table_cell_style), Paragraph("97.93%", table_cell_style), Paragraph("97.14% (± 0.68%)", table_cell_style)]
    ]
    t_comp = Table(comp_data, colWidths=[130, 65, 70, 65, 65, 90])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#e8f5e9"), colors.white, colors.white, colors.white, colors.white]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_comp)
    story.append(Paragraph("Table 2: Comparative performance evaluation across 5 benchmarked Machine Learning models.", caption_style))
    story.append(Spacer(1, 10))

    story.append(PageBreak())

    # Comparison and Feature Importance Charts
    feat_imp_img = os.path.join(EDA_DIR, "07_feature_importance.png")
    if os.path.exists(feat_imp_img):
        story.append(Image(feat_imp_img, width=5.5*inch, height=3.0*inch))
        story.append(Paragraph("Figure 3: Feature Importance Comparison (Gini Impurity Reduction: RF vs DT).", caption_style))

    story.append(Paragraph(
        "<b>Analysis of Model Comparison:</b> The Random Forest classifier demonstrated superior generalizability with 99.55% accuracy and an exceptionally tight 5-fold cross-validation standard deviation (± 0.32%). "
        "Out of 440 test samples, Random Forest misclassified only 2 instances (1 rice sample predicted as jute due to overlapping high rainfall/humidity conditions, and 1 blackgram sample). "
        "In contrast, the single Decision Tree misclassified 9 instances (accuracy 97.95%), illustrating how ensemble bagging curtails high variance and stabilizes decision boundaries.",
        body_style
    ))

    # =========================================================================
    # 12. FEATURE IMPORTANCE & LIMITATIONS
    # =========================================================================
    story.append(Paragraph("10. Feature Importance and Limitations Analysis", h1_style))
    story.append(Paragraph(
        "<b>Feature Importance Insights:</b> "
        "<br/>1. <b>Rainfall (23.02%) & Humidity (22.42%):</b> Atmospheric moisture parameters represent 45.44% of the cumulative decision weight, acting as the primary macro-split separating arid/semi-arid crops from tropical/monsoon vegetation. "
        "<br/>2. <b>Potassium (17.54%) & Phosphorus (15.09%):</b> Represent 32.63% of the decision weight, providing the critical discriminating signal for fruit orchards versus field pulses. "
        "<br/>3. <b>Nitrogen (9.64%), Temperature (7.24%), and pH (5.06%):</b> Act as fine-grained discriminators within specific crop sub-clusters.",
        body_style
    ))
    story.append(Paragraph("<b>System Limitations:</b>", h2_style))
    story.append(Paragraph("• <b>Discrete Class Scope:</b> The model is trained on 22 specific crop labels and does not yet extrapolate to unrepresented regional hybrids or specialty horticulture.", bullet_style))
    story.append(Paragraph("• <b>Static Seasonal Averages:</b> Inputs represent cumulative or average seasonal values, neglecting temporal weather extremes (e.g., sudden frost, flash floods).", bullet_style))
    story.append(Paragraph("• <b>Economic & Market Disconnect:</b> While a crop may be agronomically optimal, the model does not consider market procurement prices, transport logistics, or crop storage life.", bullet_style))
    story.append(Paragraph("• <b>Soil Physical Properties:</b> Parameters like soil depth, water infiltration rate, and drainage gradient are omitted from the current tabular feature space.", bullet_style))

    # =========================================================================
    # 13. STREAMLIT DEPLOYMENT
    # =========================================================================
    story.append(Paragraph("11. Streamlit Application Deployment and User Interface", h1_style))
    story.append(Paragraph(
        "The finalized Random Forest model was serialized to disk using `joblib` (`models/crop_recommendation_rf.pkl`) and deployed via an interactive, production-grade <b>Streamlit</b> web dashboard (`app.py`). "
        "The application architecture incorporates five functional tabs:",
        body_style
    ))
    story.append(Paragraph("• <b>Single Recommendation & Soil Diagnostic:</b> Features interactive sliders with real-time agronomic validation alerts (e.g. soil acidity warnings), multi-scenario preset buttons, top-match recommendation, top-3 ranked alternatives with confidence percentages, and an agronomic cultivation guide.", bullet_style))
    story.append(Paragraph("• <b>Batch CSV Processing:</b> Enables farmers and agricultural field officers to upload CSV files with hundreds of soil test results, producing real-time bulk predictions and downloadable output reports.", bullet_style))
    story.append(Paragraph("• <b>Interactive EDA Hub:</b> Visualizes data distributions, correlation heatmaps, and side-by-side nutritional profile comparisons between any two chosen crops.", bullet_style))
    story.append(Paragraph("• <b>Model Benchmarking:</b> Displays complete quantitative comparison tables, bar charts, confusion matrix heatmaps, and feature importance rankings.", bullet_style))
    story.append(Paragraph("• <b>Viva & Documentation Guide:</b> Built-in viva reference manual answering common faculty assessment queries.", bullet_style))

    story.append(PageBreak())

    # Add Application Screenshots
    app_hero_img = os.path.join(SCREENSHOTS_DIR, "app_hero.png")
    app_rec_img = os.path.join(SCREENSHOTS_DIR, "app_recommendation.png")
    if os.path.exists(app_hero_img) and os.path.exists(app_rec_img):
        story.append(Image(app_hero_img, width=6.0*inch, height=2.4*inch))
        story.append(Paragraph("Figure 4: AgroSense Streamlit Web Dashboard - Top Hero & KPI Cards.", caption_style))
        story.append(Spacer(1, 8))
        story.append(Image(app_rec_img, width=6.0*inch, height=2.4*inch))
        story.append(Paragraph("Figure 5: Precision Crop Recommendation Output with Top-3 Probability Scores & Cultivation Dossier.", caption_style))

    # =========================================================================
    # 14. RESULTS & DISCUSSION
    # =========================================================================
    story.append(Paragraph("12. Results and Discussion", h1_style))
    story.append(Paragraph(
        "Empirical testing revealed that precision crop recommendation is highly amenable to machine learning modeling. "
        "Because each crop species has evolved specific biochemical and photosynthetic adaptations, their ecological niches form distinct convex clusters in the 7-dimensional feature space. "
        "Random Forest successfully captured these non-linear decision boundaries, attaining 99.55% accuracy. "
        "The confusion matrix indicates that near-perfect precision and recall (> 0.97) were achieved across all 22 classes. "
        "Minor misclassifications occurred exclusively between crops sharing near-identical ecological envelopes (e.g., Rice and Jute during peak monsoon), which our top-3 ranked recommendation UI successfully mitigates by presenting both options to the farmer.",
        body_style
    ))

    # =========================================================================
    # 15. CONCLUSION & FUTURE SCOPE
    # =========================================================================
    story.append(Paragraph("13. Conclusion and Future Scope", h1_style))
    story.append(Paragraph(
        "<b>Conclusion:</b> "
        "The <b>AgroSense</b> mini-project successfully developed and deployed an end-to-end Machine Learning pipeline that fulfills all requirements specified for Group 15. "
        "By systematically progressing through data understanding, preprocessing, statistical EDA, multi-model benchmarking, and interactive deployment, we demonstrated that ensemble machine learning can accurately translate soil chemistry and weather metrics into actionable agronomic guidance.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Future Scope:</b>"
        "<br/>1. <b>IoT Soil Sensor Integration:</b> Interface the Streamlit application with ESP32-based NPK and moisture probes for automated real-time data telemetry. "
        "<br/>2. <b>Live Weather API Integration:</b> Fetch real-time localized temperature, humidity, and rainfall forecasts directly via OpenWeatherMap API based on GPS coordinates. "
        "<br/>3. <b>Market Price & Yield Forecasting:</b> Integrate wholesale commodity market APIs (e.g., e-NAM) to predict anticipated revenue per hectare alongside biological suitability. "
        "<br/>4. <b>Multilingual Voice Assistant:</b> Develop vernacular voice interfaces (Hindi, Marathi, Telugu, etc.) to bridge digital literacy barriers for rural farming communities.",
        body_style
    ))

    # =========================================================================
    # 16. INDIVIDUAL CONTRIBUTION RECORD & REFERENCES
    # =========================================================================
    story.append(Paragraph("14. Individual Contribution Record (Group 15)", h1_style))
    contrib_data = [
        [Paragraph("<b>Component / Lifecycle Phase</b>", table_header_style), Paragraph("<b>Key Responsibilities & Deliverables</b>", table_header_style), Paragraph("<b>Contribution %</b>", table_header_style)],
        [Paragraph("Problem Definition & EDA", table_cell_style), Paragraph("Formulation of objectives, dataset audit, generation of 6 publication-grade EDA visualizations.", table_cell_style), Paragraph("25%", table_cell_style)],
        [Paragraph("Data Preprocessing & Splitting", table_cell_style), Paragraph("Missing value check, outlier validation, stratified 80:20 partitioning, and feature scaling pipeline.", table_cell_style), Paragraph("25%", table_cell_style)],
        [Paragraph("ML Model Building & Tuning", table_cell_style), Paragraph("Implementation of Random Forest, Decision Tree, Naive Bayes, SVM, KNN, and 5-fold CV evaluation.", table_cell_style), Paragraph("25%", table_cell_style)],
        [Paragraph("Deployment & Technical Report", table_cell_style), Paragraph("Streamlit web app development, batch CSV predictor, report compilation, and viva preparation.", table_cell_style), Paragraph("25%", table_cell_style)]
    ]
    t_contrib = Table(contrib_data, colWidths=[130, 290, 65])
    t_contrib.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_contrib)
    story.append(Spacer(1, 10))

    story.append(Paragraph("15. References", h1_style))
    story.append(Paragraph("1. Breiman, L. (2001). <i>Random Forests</i>. Machine Learning, 45(1), 5-32.", bullet_style))
    story.append(Paragraph("2. Quinlan, J. R. (1986). <i>Induction of Decision Trees</i>. Machine Learning, 1(1), 81-106.", bullet_style))
    story.append(Paragraph("3. Bondre, D. A., & Mahagaonkar, S. (2019). <i>Prediction of Crop Yield and Fertilizer Recommendation using Machine Learning</i>. International Journal of Engineering Applied Sciences and Technology, 4(5), 371-376.", bullet_style))
    story.append(Paragraph("4. Pudumalar, S., et al. (2016). <i>Crop Recommendation System for Precision Agriculture using Machine Learning Techniques</i>. Eighth International Conference on Advanced Computing (ICoAC), 32-36.", bullet_style))
    story.append(Paragraph("5. Pedregosa, F., et al. (2011). <i>Scikit-learn: Machine Learning in Python</i>. Journal of Machine Learning Research, 12, 2825-2830.", bullet_style))
    story.append(Paragraph("6. Indian Council of Agricultural Research (ICAR). <i>Handbook of Agriculture: Facts and Figures for Farmers, Students and All Interested in Farming</i>.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✓ Project Report PDF compiled successfully: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
