from fpdf import FPDF
from src.utils import strip_accents

def generate_pdf_report(city, target_date, risk, confidence, weather_data):
    """
    Generate PDF report for flood risk assessment.
    Returns bytes of the PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(0, 10, "Vietnam Flood Risk Assessment Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Location: {strip_accents(city)}", ln=True)
    pdf.cell(0, 10, f"Analysis Date: {target_date.strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(0, 10, f"Risk Level: {risk}", ln=True)
    pdf.cell(0, 10, f"AI Confidence: {confidence:.1f}%", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Weather Conditions:", ln=True)
    pdf.cell(0, 10, f"  - Max Temperature: {weather_data['max']:.1f} deg C", ln=True)
    pdf.cell(0, 10, f"  - Min Temperature: {weather_data['min']:.1f} deg C", ln=True)
    pdf.cell(0, 10, f"  - Humidity: {weather_data['humidi']}%", ln=True)
    pdf.cell(0, 10, f"  - Rainfall (24h): {weather_data['rain']:.1f} mm", ln=True)
    pdf.cell(0, 10, f"  - Rainfall (72h): {weather_data['rain_last_3_days']:.1f} mm", ln=True)
    pdf.cell(0, 10, f"  - Wind Speed: {weather_data['wind']:.1f} m/s", ln=True)
    pdf.cell(0, 10, f"  - Pressure: {weather_data['pressure']} hPa", ln=True)
    pdf.cell(0, 10, f"  - 7-Day Rain Total: {weather_data['rain_last_7_days']:.1f} mm", ln=True)
    pdf.ln(5)
    # Soil Moisture Analysis
    saturation_level = min(weather_data['rain_last_3_days'] / 150.0, 1.0)
    saturation_percentage = saturation_level * 100
    pdf.cell(0, 10, "Soil Moisture Analysis:", ln=True)
    pdf.cell(0, 10, f"  - Ground Saturation: {saturation_percentage:.1f}% (based on 72h rainfall)", ln=True)
    if saturation_level > 0.8:
        pdf.cell(0, 10, "  - WARNING: Soil is highly saturated. Flood risk is significantly increased.", ln=True)
    elif saturation_level > 0.5:
        pdf.cell(0, 10, "  - ALERT: Moderate soil saturation. Monitor for potential runoff.", ln=True)
    else:
        pdf.cell(0, 10, "  - Ground conditions appear normal.", ln=True)
    # Add visual bar for soil saturation
    pdf.ln(5)
    pdf.cell(0, 10, "Soil Saturation Visual:", ln=True)
    # Draw background bar
    pdf.set_fill_color(200, 200, 200)  # Light gray
    pdf.rect(10, pdf.get_y(), 180, 10, 'F')
    # Draw filled bar based on saturation
    if saturation_level > 0.8:
        pdf.set_fill_color(255, 0, 0)  # Red for high
    elif saturation_level > 0.5:
        pdf.set_fill_color(255, 165, 0)  # Orange for medium
    else:
        pdf.set_fill_color(0, 128, 0)  # Green for low
    pdf.rect(10, pdf.get_y(), 180 * saturation_level, 10, 'F')
    pdf.set_fill_color(0, 0, 0)  # Reset to black
    pdf.ln(15)
    pdf.ln(5)
    pdf.cell(0, 10, f"Risk Assessment: {risk} flood risk detected.", ln=True)
    if risk == 'High':
        pdf.cell(0, 10, "Recommendation: Prepare for potential flooding. Monitor weather updates.", ln=True)
    elif risk == 'Medium':
        pdf.cell(0, 10, "Recommendation: Stay alert for localized flooding in low-lying areas.", ln=True)
    else:
        pdf.cell(0, 10, "Recommendation: Conditions appear stable.", ln=True)

    return bytes(pdf.output())
