#!/usr/bin/env python3
"""
Comprehensive Report Generation for Phase 1: Material Characterization
Fire-Resistant Rubberized Concrete Project
"""

import json
import os
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfgen import canvas

# Define paths
DATA_DIR = Path('../data')
REPORTS_DIR = Path('../reports')
VIZ_DIR = Path('../visualizations')
REPORTS_DIR.mkdir(exist_ok=True)

class MaterialCharacterizationReport:
    def __init__(self):
        self.data = self.load_all_data()
        self.styles = getSampleStyleSheet()
        self.create_custom_styles()
        self.story = []
        
    def load_all_data(self):
        """Load all JSON data files"""
        data = {}
        data['cement'] = json.load(open(DATA_DIR / 'cement_properties.json'))
        data['aggregates'] = json.load(open(DATA_DIR / 'aggregates_properties.json'))
        data['rubber'] = json.load(open(DATA_DIR / 'crumb_rubber_properties.json'))
        data['water_admix'] = json.load(open(DATA_DIR / 'water_and_admixtures.json'))
        data['mix_designs'] = json.load(open(DATA_DIR / 'mix_designs_matrix.json'))
        data['fresh_props'] = json.load(open(DATA_DIR / 'fresh_concrete_properties.json'))
        return data
    
    def create_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#2E4057'),
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#048A81'),
            spaceBefore=20,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#54C6EB'),
            spaceBefore=15,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='Justify',
            parent=self.styles['BodyText'],
            alignment=TA_JUSTIFY,
            fontSize=11,
            leading=14
        ))
    
    def add_title_page(self):
        """Add title page"""
        self.story.append(Spacer(1, 2*inch))
        
        title = Paragraph(
            "PHASE 1: MATERIAL CHARACTERIZATION<br/>& SPECIMEN PREPARATION",
            self.styles['CustomTitle']
        )
        self.story.append(title)
        
        self.story.append(Spacer(1, 0.5*inch))
        
        subtitle = Paragraph(
            "Development and Validation of a Thermo-Mechanical Model for<br/>"
            "Fire-Resistant Structural Elements Utilizing<br/>"
            "High-Performance Rubberized Concrete",
            self.styles['Title']
        )
        self.story.append(subtitle)
        
        self.story.append(Spacer(1, 1*inch))
        
        info = Paragraph(
            f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y')}<br/>"
            f"<b>Laboratory:</b> Advanced Concrete Technology Lab<br/>"
            f"<b>Principal Investigator:</b> Dr. Sarah Chen<br/>"
            f"<b>Test Standards:</b> ASTM C150, C136, C143, C231, C138, C403",
            self.styles['Normal']
        )
        self.story.append(info)
        
        self.story.append(PageBreak())
    
    def add_executive_summary(self):
        """Add executive summary"""
        self.story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        
        summary_text = """
        This comprehensive report presents the complete material characterization and mix design 
        development for fire-resistant rubberized concrete. The study encompasses detailed analysis 
        of all constituent materials including Type I Portland cement, natural aggregates, crumb 
        rubber from end-of-life tires, and chemical admixtures.
        <br/><br/>
        A total of 12 concrete mix designs were developed with rubber replacement levels ranging 
        from 0% to 20% by volume of fine aggregate. Both fine (1-4mm) and coarse (4-8mm) rubber 
        particles were investigated, along with silica fume modified mixtures for enhanced performance.
        <br/><br/>
        Key findings from the material characterization phase include:
        <br/>
        • Cement exhibits normal Type I characteristics with 58.2% C3S content
        <br/>
        • Aggregates meet ASTM gradation requirements with FM = 2.72 for sand
        <br/>
        • Crumb rubber shows hydrophobic nature with 118° contact angle
        <br/>
        • Fresh properties indicate workability reduction with increased rubber content
        <br/>
        • Air content increases from 2.1% to 5.2% with rubber addition
        <br/>
        • Unit weight reduces by up to 7.7% at 20% rubber replacement
        """
        
        self.story.append(Paragraph(summary_text, self.styles['Justify']))
        self.story.append(PageBreak())
    
    def add_cement_section(self):
        """Add cement characterization section"""
        self.story.append(Paragraph("1. CEMENT CHARACTERIZATION", self.styles['SectionHeader']))
        
        cement_data = self.data['cement']
        
        # Add description
        intro = Paragraph(
            f"The cement used is {cement_data['cement_type']} manufactured by {cement_data['manufacturer']}. "
            f"Comprehensive testing was performed including XRF chemical analysis, XRD mineralogical analysis, "
            f"and physical property determination.",
            self.styles['Justify']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Chemical composition table
        self.story.append(Paragraph("1.1 Chemical Composition (XRF Analysis)", self.styles['SubsectionHeader']))
        
        chem_data = []
        chem_data.append(['Oxide', 'Content (%)'])
        for oxide, value in cement_data['chemical_composition_xrf'].items():
            if oxide not in ['total', 'free_CaO'] and value > 0.1:
                chem_data.append([oxide, f"{value:.2f}"])
        
        chem_table = Table(chem_data, colWidths=[2*inch, 1.5*inch])
        chem_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(chem_table)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Bogue composition
        self.story.append(Paragraph("1.2 Bogue Composition", self.styles['SubsectionHeader']))
        
        bogue_data = []
        bogue_data.append(['Phase', 'Content (%)'])
        for phase, value in cement_data['bogue_composition'].items():
            if phase != 'calculation_method':
                bogue_data.append([phase, f"{value:.1f}"])
        
        bogue_table = Table(bogue_data, colWidths=[2*inch, 1.5*inch])
        bogue_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(bogue_table)
        
        # Add visualization if exists
        if (VIZ_DIR / 'cement_characterization.png').exists():
            self.story.append(Spacer(1, 0.3*inch))
            img = Image(str(VIZ_DIR / 'cement_characterization.png'), width=6*inch, height=4.5*inch)
            self.story.append(img)
        
        self.story.append(PageBreak())
    
    def add_aggregates_section(self):
        """Add aggregates characterization section"""
        self.story.append(Paragraph("2. AGGREGATES CHARACTERIZATION", self.styles['SectionHeader']))
        
        agg_data = self.data['aggregates']
        
        # Coarse aggregate
        self.story.append(Paragraph("2.1 Coarse Aggregate Properties", self.styles['SubsectionHeader']))
        
        coarse = agg_data['coarse_aggregate']
        coarse_props = [
            ['Property', 'Value', 'Unit'],
            ['Type', coarse['type'], '-'],
            ['Specific Gravity (SSD)', f"{coarse['physical_properties']['specific_gravity']['bulk_ssd']:.2f}", '-'],
            ['Water Absorption', f"{coarse['physical_properties']['water_absorption']['value']:.2f}", '%'],
            ['Bulk Density (Loose)', f"{coarse['physical_properties']['bulk_density']['loose']['value']:.0f}", 'kg/m³'],
            ['LA Abrasion', f"{coarse['physical_properties']['los_angeles_abrasion']['value']:.1f}", '%'],
            ['Nominal Max Size', f"{coarse['sieve_analysis']['nominal_max_size_mm']:.0f}", 'mm']
        ]
        
        coarse_table = Table(coarse_props, colWidths=[2*inch, 1.5*inch, 1*inch])
        coarse_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(coarse_table)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Fine aggregate
        self.story.append(Paragraph("2.2 Fine Aggregate Properties", self.styles['SubsectionHeader']))
        
        fine = agg_data['fine_aggregate']
        fine_props = [
            ['Property', 'Value', 'Unit'],
            ['Type', fine['type'], '-'],
            ['Specific Gravity (SSD)', f"{fine['physical_properties']['specific_gravity']['bulk_ssd']:.2f}", '-'],
            ['Water Absorption', f"{fine['physical_properties']['water_absorption']['value']:.2f}", '%'],
            ['Bulk Density (Loose)', f"{fine['physical_properties']['bulk_density']['loose']['value']:.0f}", 'kg/m³'],
            ['Fineness Modulus', f"{fine['sieve_analysis']['fineness_modulus']:.2f}", '-'],
            ['Grading Zone', fine['sieve_analysis']['grading_zone'], '-']
        ]
        
        fine_table = Table(fine_props, colWidths=[2*inch, 1.5*inch, 1*inch])
        fine_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(fine_table)
        
        # Add gradation plot if exists
        if (VIZ_DIR / 'aggregate_gradation.png').exists():
            self.story.append(Spacer(1, 0.3*inch))
            img = Image(str(VIZ_DIR / 'aggregate_gradation.png'), width=6*inch, height=3*inch)
            self.story.append(img)
        
        self.story.append(PageBreak())
    
    def add_rubber_section(self):
        """Add crumb rubber characterization section"""
        self.story.append(Paragraph("3. CRUMB RUBBER CHARACTERIZATION", self.styles['SectionHeader']))
        
        rubber_data = self.data['rubber']['crumb_rubber_characterization']
        
        intro = Paragraph(
            f"Crumb rubber was sourced from {rubber_data['source']['type']} processed by "
            f"{rubber_data['source']['processing_method']}. Two size ranges were prepared: "
            f"Fine (1-4mm) and Coarse (4-8mm) for comparative evaluation.",
            self.styles['Justify']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Physical properties
        self.story.append(Paragraph("3.1 Physical Properties", self.styles['SubsectionHeader']))
        
        phys_props = [
            ['Property', 'Value', 'Unit'],
            ['Specific Gravity', f"{rubber_data['physical_properties']['specific_gravity']:.2f}", '-'],
            ['Bulk Density (Loose)', f"{rubber_data['physical_properties']['bulk_density']['loose']['value']:.0f}", 'kg/m³'],
            ['Mohs Hardness', f"{rubber_data['physical_properties']['mohs_hardness']:.1f}", '-'],
            ['Porosity', f"{rubber_data['physical_properties']['porosity']['value']:.1f}", '%'],
            ['Moisture Content', f"{rubber_data['physical_properties']['moisture_content']['value']:.1f}", '%']
        ]
        
        phys_table = Table(phys_props, colWidths=[2*inch, 1.5*inch, 1*inch])
        phys_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(phys_table)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Surface properties
        self.story.append(Paragraph("3.2 Surface Properties", self.styles['SubsectionHeader']))
        
        surface_props = [
            ['Property', 'Value', 'Unit'],
            ['Surface Area (BET)', f"{rubber_data['surface_properties']['surface_area_bet']['value']:.3f}", 'm²/g'],
            ['Water Contact Angle', f"{rubber_data['surface_properties']['water_contact_angle']['value']:.0f}", '°'],
            ['Surface Energy (Total)', f"{rubber_data['surface_properties']['surface_energy']['total']:.1f}", 'mJ/m²'],
            ['Classification', rubber_data['surface_properties']['water_contact_angle']['classification'], '-']
        ]
        
        surface_table = Table(surface_props, colWidths=[2*inch, 1.5*inch, 1*inch])
        surface_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(surface_table)
        
        # Add rubber characterization plot if exists
        if (VIZ_DIR / 'rubber_characterization.png').exists():
            self.story.append(Spacer(1, 0.3*inch))
            img = Image(str(VIZ_DIR / 'rubber_characterization.png'), width=6*inch, height=4*inch)
            self.story.append(img)
        
        self.story.append(PageBreak())
    
    def add_mix_design_section(self):
        """Add mix design section"""
        self.story.append(Paragraph("4. MIX DESIGN MATRIX", self.styles['SectionHeader']))
        
        mix_data = self.data['mix_designs']['mix_designs']
        
        intro = Paragraph(
            "A comprehensive matrix of 12 mix designs was developed to evaluate the effect of rubber "
            "content (0-20%), particle size (fine vs coarse), and silica fume addition on concrete properties. "
            "All mixes were designed for a target 28-day strength of 40 MPa with W/C ratio of approximately 0.45.",
            self.styles['Justify']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Create mix design summary table
        mix_summary = [['Mix ID', 'Rubber\n(%)', 'Size', 'Cement\n(kg/m³)', 'Water\n(kg/m³)', 'W/C']]
        
        for mix_id, mix_info in mix_data.items():
            rubber_pct = mix_info['rubber_replacement']['percentage']
            if rubber_pct == 0:
                size = '-'
            elif 'Fine' in mix_info['designation']:
                size = 'Fine'
            elif 'Coarse' in mix_info['designation']:
                size = 'Coarse'
            else:
                size = 'Mixed'
            
            mix_summary.append([
                mix_id,
                f"{rubber_pct}",
                size,
                f"{mix_info['proportions_kg_m3']['cement']:.0f}",
                f"{mix_info['proportions_kg_m3']['water']:.0f}",
                f"{mix_info['actual_wc_ratio']:.3f}"
            ])
        
        mix_table = Table(mix_summary, colWidths=[0.8*inch, 0.7*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.7*inch])
        mix_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        self.story.append(mix_table)
        
        # Add mix design analysis plot if exists
        if (VIZ_DIR / 'mix_design_analysis.png').exists():
            self.story.append(Spacer(1, 0.3*inch))
            img = Image(str(VIZ_DIR / 'mix_design_analysis.png'), width=6*inch, height=4.5*inch)
            self.story.append(img)
        
        self.story.append(PageBreak())
    
    def add_fresh_properties_section(self):
        """Add fresh concrete properties section"""
        self.story.append(Paragraph("5. FRESH CONCRETE PROPERTIES", self.styles['SectionHeader']))
        
        fresh_data = self.data['fresh_props']['fresh_properties_results']
        
        intro = Paragraph(
            "Fresh concrete properties were evaluated immediately after mixing to assess workability, "
            "air content, density, and rheological behavior. Tests were conducted according to ASTM standards "
            "under controlled laboratory conditions (23±2°C, 65±5% RH).",
            self.styles['Justify']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Create fresh properties summary table
        fresh_summary = [['Mix ID', 'Slump\n(mm)', 'Air\n(%)', 'Density\n(kg/m³)', 'Initial Set\n(min)']]
        
        for mix_id, props in fresh_data.items():
            fresh_summary.append([
                mix_id,
                f"{props['slump']['initial']['value']:.0f}",
                f"{props['air_content']['value']:.1f}",
                f"{props['unit_weight']['value']:.0f}",
                f"{props['setting_time']['initial']['value']:.0f}"
            ])
        
        fresh_table = Table(fresh_summary, colWidths=[1*inch, 0.9*inch, 0.7*inch, 1*inch, 1*inch])
        fresh_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        self.story.append(fresh_table)
        
        # Add fresh properties analysis plot if exists
        if (VIZ_DIR / 'fresh_properties_analysis.png').exists():
            self.story.append(Spacer(1, 0.3*inch))
            img = Image(str(VIZ_DIR / 'fresh_properties_analysis.png'), width=6*inch, height=4*inch)
            self.story.append(img)
        
        self.story.append(PageBreak())
    
    def add_conclusions(self):
        """Add conclusions section"""
        self.story.append(Paragraph("6. CONCLUSIONS AND KEY FINDINGS", self.styles['SectionHeader']))
        
        conclusions = """
        The comprehensive material characterization and mix design development phase has successfully 
        established the baseline data for fire-resistant rubberized concrete development. Key findings include:
        <br/><br/>
        <b>Material Characteristics:</b><br/>
        • Type I Portland cement shows typical composition with adequate strength development<br/>
        • Natural aggregates meet all ASTM requirements for structural concrete<br/>
        • Crumb rubber exhibits hydrophobic nature requiring increased admixture dosage<br/>
        • Two distinct rubber size ranges provide different fresh and hardened properties<br/>
        <br/>
        <b>Mix Design Performance:</b><br/>
        • Workability decreases progressively with rubber content (180mm to 115mm slump)<br/>
        • Air content increases from 2.1% (control) to 5.2% (20% rubber)<br/>
        • Density reduction of 7.7% achieved at maximum rubber replacement<br/>
        • Setting time delayed by 15-20% with rubber addition<br/>
        • Superplasticizer demand increases exponentially with rubber content<br/>
        <br/>
        <b>Optimization Observations:</b><br/>
        • Fine rubber (1-4mm) provides better workability than coarse rubber<br/>
        • Silica fume addition partially compensates for strength loss<br/>
        • 10-15% rubber replacement appears optimal for balanced properties<br/>
        • Rheological properties show critical threshold at 15% replacement<br/>
        <br/>
        <b>Recommendations for Phase 2:</b><br/>
        • Focus on 10% and 15% rubber replacement for detailed investigation<br/>
        • Implement surface treatment techniques for rubber particles<br/>
        • Evaluate high-temperature performance up to 800°C<br/>
        • Conduct microstructural analysis of interfacial transition zone<br/>
        • Develop thermal-mechanical constitutive models
        """
        
        self.story.append(Paragraph(conclusions, self.styles['Justify']))
        self.story.append(PageBreak())
    
    def add_appendix(self):
        """Add appendix with additional data"""
        self.story.append(Paragraph("APPENDIX A: TEST STANDARDS AND METHODS", self.styles['SectionHeader']))
        
        standards = [
            ['Test', 'Standard', 'Description'],
            ['Cement Composition', 'ASTM C114', 'Chemical Analysis of Hydraulic Cement'],
            ['Cement Fineness', 'ASTM C204', 'Blaine Air Permeability'],
            ['Aggregate Gradation', 'ASTM C136', 'Sieve Analysis'],
            ['Aggregate Specific Gravity', 'ASTM C127/128', 'Coarse and Fine Aggregates'],
            ['Slump', 'ASTM C143', 'Slump of Hydraulic-Cement Concrete'],
            ['Air Content', 'ASTM C231', 'Pressure Method'],
            ['Unit Weight', 'ASTM C138', 'Density of Fresh Concrete'],
            ['Setting Time', 'ASTM C403', 'Penetration Resistance'],
            ['Temperature', 'ASTM C1064', 'Temperature of Fresh Concrete'],
            ['Rubber Analysis', 'ASTM D5603', 'Rubber from Recycled Tires']
        ]
        
        standards_table = Table(standards, colWidths=[1.5*inch, 1.2*inch, 3*inch])
        standards_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        self.story.append(standards_table)
    
    def generate_report(self):
        """Generate the complete PDF report"""
        output_file = REPORTS_DIR / f"Phase1_Material_Characterization_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        doc = SimpleDocTemplate(
            str(output_file),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build the report
        self.add_title_page()
        self.add_executive_summary()
        self.add_cement_section()
        self.add_aggregates_section()
        self.add_rubber_section()
        self.add_mix_design_section()
        self.add_fresh_properties_section()
        self.add_conclusions()
        self.add_appendix()
        
        # Generate PDF
        doc.build(self.story)
        
        return output_file

def main():
    """Main execution function"""
    print("Generating comprehensive Phase 1 report...")
    
    report_generator = MaterialCharacterizationReport()
    output_file = report_generator.generate_report()
    
    print(f"Report successfully generated: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")

if __name__ == "__main__":
    main()