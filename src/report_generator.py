"""
WPW ECG Detection - PDF Report Generation Module

This module generates professional, clinical-grade PDF reports for WPW syndrome
detection results, including ECG analysis, AI predictions, and medical recommendations.

Features:
- Professional medical report formatting
- Patient information and demographics
- ECG feature analysis tables
- AI prediction results with risk classification
- Clinical recommendations
- AI-generated medical explanations
- Compliance-ready documentation

Author: Enhanced with Claude Code
Date: 2026
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, PageBreak, Image
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: ReportLab not installed. Install with: pip install reportlab")


class WPWReportGenerator:
    """
    Professional PDF report generator for WPW ECG analysis.

    Generates comprehensive clinical reports including patient data,
    ECG features, AI predictions, and medical recommendations.
    """

    def __init__(self, output_dir: str = "../reports"):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save generated reports
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab not installed. Install with: pip install reportlab"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#666666'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))

        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))

        # Clinical finding style
        self.styles.add(ParagraphStyle(
            name='ClinicalFinding',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))

    def generate_report(
        self,
        patient_info: Dict,
        features: Dict,
        prediction_result: Dict,
        llm_explanation: Optional[str] = None,
        model_name: str = "AI Detection System",
        doctor_name: str = "Reviewing Physician",
        hospital_name: str = "Medical Center"
    ) -> str:
        """
        Generate comprehensive WPW ECG analysis PDF report.

        Args:
            patient_info: Patient demographics (name, id, age, sex, etc.)
            features: ECG feature dictionary
            prediction_result: Prediction results from predictor
            llm_explanation: Optional AI-generated medical explanation
            model_name: Name of ML model used
            doctor_name: Reviewing physician name
            hospital_name: Institution name

        Returns:
            Path to generated PDF file
        """
        # Generate filename
        patient_id = patient_info.get('id', 'UNKNOWN')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"wpw_ecg_report_{patient_id}_{timestamp}.pdf"
        filepath = self.output_dir / filename

        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Build content
        story = []

        # Header
        story.extend(self._create_header(hospital_name))

        # Title
        story.append(Paragraph(
            "WPW SYNDROME ECG ANALYSIS REPORT",
            self.styles['ReportTitle']
        ))

        story.append(Paragraph(
            f"AI-Assisted ECG Screening · {datetime.now().strftime('%B %d, %Y')}",
            self.styles['ReportSubtitle']
        ))

        story.append(Spacer(1, 0.3*inch))

        # Patient Information
        story.extend(self._create_patient_info_section(patient_info))

        # Prediction Results (Prominent Box)
        story.extend(self._create_prediction_results_section(prediction_result, model_name))

        # ECG Features Table
        story.extend(self._create_ecg_features_section(features))

        # AI Medical Analysis
        if llm_explanation:
            story.extend(self._create_ai_analysis_section(llm_explanation))

        # Clinical Recommendations
        story.extend(self._create_recommendations_section(prediction_result))

        # Footer
        story.extend(self._create_footer(doctor_name))

        # Build PDF
        doc.build(story)

        print(f"Report generated: {filepath}")
        return str(filepath)

    def _create_header(self, hospital_name: str) -> list:
        """Create report header."""
        elements = []

        header_data = [[
            Paragraph(f"<b>{hospital_name}</b><br/>Cardiology Department<br/>ECG Analysis Unit",
                     self.styles['Normal'])
        ]]

        header_table = Table(header_data, colWidths=[6.5*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))

        elements.append(header_table)
        elements.append(Spacer(1, 0.2*inch))

        return elements

    def _create_patient_info_section(self, patient_info: Dict) -> list:
        """Create patient information section."""
        elements = []

        elements.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeading']))

        # Patient data table
        data = [
            ['Patient Name:', patient_info.get('name', 'N/A'),
             'Patient ID:', patient_info.get('id', 'N/A')],
            ['Age:', f"{patient_info.get('age', 'N/A')} years",
             'Sex:', patient_info.get('sex', 'N/A')],
            ['Date of ECG:', patient_info.get('ecg_date', datetime.now().strftime('%Y-%m-%d')),
             'Report Date:', datetime.now().strftime('%Y-%m-%d')]
        ]

        table = Table(data, colWidths=[1.5*inch, 1.75*inch, 1.5*inch, 1.75*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _create_prediction_results_section(self, prediction_result: Dict, model_name: str) -> list:
        """Create prominent prediction results section."""
        elements = []

        elements.append(Paragraph("AI PREDICTION RESULTS", self.styles['SectionHeading']))

        # Determine colors based on prediction
        prediction = prediction_result.get('prediction', 0)
        wpw_prob = prediction_result.get('wpw_probability', 0)
        risk_level = prediction_result.get('risk_level', 'unknown').upper()

        if prediction == 1:
            result_color = colors.HexColor('#d32f2f')  # Red for WPW detected
            result_text = "⚠ WPW SYNDROME PATTERN DETECTED"
        else:
            result_color = colors.HexColor('#388e3c')  # Green for normal
            result_text = "✓ NORMAL SINUS RHYTHM"

        # Risk level colors
        risk_colors = {
            'LOW': colors.HexColor('#4caf50'),
            'MODERATE': colors.HexColor('#ff9800'),
            'HIGH': colors.HexColor('#f44336'),
            'VERY_HIGH': colors.HexColor('#b71c1c')
        }
        risk_color = risk_colors.get(risk_level, colors.grey)

        # Results table
        data = [
            [Paragraph(f"<b>{result_text}</b>", self.styles['Normal'])],
            [Paragraph(f"<b>WPW Probability:</b> {wpw_prob:.1%}", self.styles['Normal'])],
            [Paragraph(f"<b>Risk Classification:</b> {risk_level}", self.styles['Normal'])],
            [Paragraph(f"<b>Confidence Level:</b> {prediction_result.get('confidence', 0):.1%}",
                      self.styles['Normal'])],
            [Paragraph(f"<b>Analysis Model:</b> {model_name}", self.styles['Normal'])]
        ]

        table = Table(data, colWidths=[6.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), result_color),
            ('BACKGROUND', (0, 2), (0, 2), risk_color),
            ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#f5f5f5')),
            ('BACKGROUND', (0, 3), (0, 4), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
            ('TEXTCOLOR', (0, 2), (0, 2), colors.white),
            ('TEXTCOLOR', (0, 1), (0, 1), colors.black),
            ('TEXTCOLOR', (0, 3), (0, 4), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 14),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _create_ecg_features_section(self, features: Dict) -> list:
        """Create ECG features analysis table."""
        elements = []

        elements.append(Paragraph("ECG FEATURE ANALYSIS", self.styles['SectionHeading']))

        # Organize features into categories
        sex_str = 'Male' if features.get('sex', 1) == 1 else 'Female'

        data = [
            ['Parameter', 'Value', 'Clinical Significance'],
            ['', '', ''],  # Header separator
            ['Demographics', '', ''],
            ['Age', f"{features.get('age', 'N/A')} years", 'Patient age'],
            ['Sex', sex_str, 'Biological sex'],
            ['', '', ''],
            ['Heart Rate Parameters', '', ''],
            ['Mean Heart Rate', f"{features.get('hr_mean', 'N/A'):.1f} bpm", 'Average heart rate'],
            ['HR Std Deviation', f"{features.get('hr_std', 'N/A'):.2f}", 'Heart rate variability'],
            ['Maximum HR', f"{features.get('hr_max', 'N/A'):.1f} bpm", 'Peak heart rate'],
            ['', '', ''],
            ['Heart Rate Variability', '', ''],
            ['HRV SDNN', f"{features.get('hrv_sdnn', 'N/A'):.4f}", 'NN interval standard deviation'],
            ['HRV RMSSD', f"{features.get('hrv_rmssd', 'N/A'):.4f}", 'Successive difference RMS'],
            ['HRV pNN50', f"{features.get('hrv_pnn50', 'N/A'):.1f}%", 'NN intervals >50ms diff'],
            ['', '', ''],
            ['QRS Complex Analysis', '', ''],
            ['QRS Duration', f"{features.get('qrs_duration_ms', 'N/A'):.2f} ms", 'Ventricular depolarization time'],
            ['QRS Amplitude', f"{features.get('qrs_amplitude', 'N/A'):.2f}", 'QRS voltage amplitude'],
            ['', '', ''],
            ['Signal Characteristics', '', ''],
            ['Signal Range', f"{features.get('signal_range', 'N/A'):.0f}", 'ECG signal dynamic range'],
            ['Signal Std Dev', f"{features.get('signal_std', 'N/A'):.2f}", 'Signal variability']
        ]

        table = Table(data, colWidths=[2.2*inch, 1.8*inch, 2.5*inch])
        table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

            # Category headers (rows 2, 6, 10, 14, 18)
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#e3f2fd')),
            ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor('#e3f2fd')),
            ('BACKGROUND', (0, 10), (-1, 10), colors.HexColor('#e3f2fd')),
            ('BACKGROUND', (0, 14), (-1, 14), colors.HexColor('#e3f2fd')),
            ('BACKGROUND', (0, 18), (-1, 18), colors.HexColor('#e3f2fd')),
            ('FONTNAME', (0, 2), (0, 2), 'Helvetica-Bold'),
            ('FONTNAME', (0, 6), (0, 6), 'Helvetica-Bold'),
            ('FONTNAME', (0, 10), (0, 10), 'Helvetica-Bold'),
            ('FONTNAME', (0, 14), (0, 14), 'Helvetica-Bold'),
            ('FONTNAME', (0, 18), (0, 18), 'Helvetica-Bold'),
            ('SPAN', (0, 2), (-1, 2)),
            ('SPAN', (0, 6), (-1, 6)),
            ('SPAN', (0, 10), (-1, 10)),
            ('SPAN', (0, 14), (-1, 14)),
            ('SPAN', (0, 18), (-1, 18)),

            # General styling
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _create_ai_analysis_section(self, llm_explanation: str) -> list:
        """Create AI-generated medical analysis section."""
        elements = []

        elements.append(Paragraph("AI-POWERED MEDICAL ANALYSIS", self.styles['SectionHeading']))

        # Clean and format the explanation
        paragraphs = llm_explanation.split('\n\n')
        for para in paragraphs:
            if para.strip():
                elements.append(Paragraph(para.strip(), self.styles['ClinicalFinding']))
                elements.append(Spacer(1, 0.1*inch))

        elements.append(Spacer(1, 0.2*inch))

        return elements

    def _create_recommendations_section(self, prediction_result: Dict) -> list:
        """Create clinical recommendations section."""
        elements = []

        elements.append(Paragraph("CLINICAL RECOMMENDATIONS", self.styles['SectionHeading']))

        recommendation = prediction_result.get('recommendation', 'Clinical correlation required.')

        elements.append(Paragraph(recommendation, self.styles['ClinicalFinding']))
        elements.append(Spacer(1, 0.2*inch))

        # Add disclaimer
        disclaimer = """
        <b>IMPORTANT CLINICAL NOTE:</b><br/>
        This report is generated by an AI-assisted screening system and is intended to support,
        not replace, clinical judgment. All findings require confirmation through standard 12-lead
        ECG interpretation by a qualified physician. This system is a screening tool and should
        be used in conjunction with comprehensive clinical assessment including patient history,
        physical examination, and additional diagnostic testing as indicated.
        """

        elements.append(Paragraph(disclaimer, self.styles['ClinicalFinding']))

        return elements

    def _create_footer(self, doctor_name: str) -> list:
        """Create report footer with signature section."""
        elements = []

        elements.append(Spacer(1, 0.4*inch))

        footer_data = [
            ['', ''],
            ['', ''],
            ['_' * 40, ''],
            [f'Reviewed by: {doctor_name}', f'Date: {datetime.now().strftime("%Y-%m-%d")}'],
            ['Cardiologist / Electrophysiologist', '']
        ]

        footer_table = Table(footer_data, colWidths=[4*inch, 2.5*inch])
        footer_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.grey),
        ]))

        elements.append(footer_table)

        # Report ID footer
        report_id = f"Report ID: WPW-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(
            f"<i>{report_id} · Generated by AI-Assisted ECG Analysis System</i>",
            self.styles['Normal']
        ))

        return elements


def main():
    """Example usage of the report generator."""
    generator = WPWReportGenerator()

    # Example data
    patient_info = {
        'name': 'John Doe',
        'id': 'P123456',
        'age': 45,
        'sex': 'Male',
        'ecg_date': '2026-11-20'
    }

    features = {
        'age': 45,
        'sex': 1,
        'hr_mean': 128.79,
        'hr_std': 23.57,
        'hr_max': 148.51,
        'hrv_sdnn': 0.129,
        'hrv_rmssd': 0.189,
        'hrv_pnn50': 35.0,
        'qrs_duration_ms': 29.62,
        'qrs_amplitude': 85.43,
        'signal_range': 4304,
        'signal_std': 256.25
    }

    prediction_result = {
        'prediction': 1,
        'interpretation': 'WPW Syndrome Pattern Detected',
        'wpw_probability': 0.87,
        'risk_level': 'high',
        'confidence': 0.74,
        'recommendation': 'Cardiology consultation recommended for comprehensive ECG review and risk stratification.'
    }

    llm_explanation = """This ECG analysis suggests possible WPW syndrome pattern based on automated feature extraction..."""

    # Generate report
    report_path = generator.generate_report(
        patient_info=patient_info,
        features=features,
        prediction_result=prediction_result,
        llm_explanation=llm_explanation,
        model_name="Random Forest Classifier",
        doctor_name="Dr. Smith",
        hospital_name="University Medical Center"
    )

    print(f"\n✅ Report generated successfully: {report_path}")


if __name__ == "__main__":
    main()
