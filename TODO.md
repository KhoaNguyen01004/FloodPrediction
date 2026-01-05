# TODO: Fix Streamlit App Issues

## 1. Update Dependencies
- [ ] Change `fpdf` to `fpdf2` in requirements.txt for UTF-8 support in PDFs
- [ ] Add `numpy` to requirements.txt for probability clipping

## 2. Fix PDF Encoding Error
- [ ] Import `FPDF` from `fpdf2` instead of `fpdf`
- [ ] Remove `.encode('latin-1')` from PDF output, as fpdf2 supports UTF-8

## 3. Fix Physical Impossibility in Temp Diff
- [ ] Change `weather_data['temp_diff'] = weather_data['max'] - weather_data['min']` to `abs(weather_data['max'] - weather_data['min'])`

## 4. Fix Probability Boosting Logic
- [ ] After probability adjustments, use `np.clip(proba, 0, 1)` and `proba /= proba.sum()` to normalize

## 5. Add Automatic Analysis on City Change
- [ ] Extract analysis code into a `run_analysis()` function
- [ ] Call `run_analysis()` when "Analyze & Predict Risk" button is pressed or when city changes

## 6. Wrap Coordinate Lookup in Try-Except
- [ ] Use `next((item for item in coords_data if item["location"] == city), None)` and handle if None

## 7. Fix Slider Freezing Issue
- [ ] Use `key` parameter for sliders (e.g., `key='sim_rain'`) instead of manual assignment to session state
- [ ] Remove lines like `st.session_state.sim_rain = st.slider(...)`
- [ ] Ensure scenario presets update session state correctly

## 8. Test the Fixes
- [ ] Run the app and verify PDF downloads work with Vietnamese characters
- [ ] Check temp_diff is always positive
- [ ] Verify probabilities sum to 1 and are between 0 and 1
- [ ] Confirm analysis runs automatically on city change
- [ ] Ensure sliders update properly and don't freeze
- [ ] Test coordinate lookup error handling
