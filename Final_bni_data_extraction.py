import pandas as pd
import numpy as np
import os, re
from collections import defaultdict
from sklearn.cluster import DBSCAN
from datetime import datetime
import warnings
from paddleocr import PaddleOCR
import cv2
import tempfile
from difflib import SequenceMatcher
import tempfile

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_mkldnn"] = "0"



# ========= CONFIG =========
COLUMNS = ['Name', 'Payment', 'TOA', 'Mode']  # Only 4 columns as requested

def simple_preprocess(image_path):
    """Enhanced preprocessing for better OCR"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Light bilateral filtering
    filtered = cv2.bilateralFilter(image, 5, 50, 50)

    # Slight sharpening for handwritten text
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(filtered, -1, kernel)

    return sharpened


def save_temp_image(image_array):
    """Save image to temporary file"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cv2.imwrite(temp_file.name, image_array)
    return temp_file.name

from difflib import SequenceMatcher


def extract_data_from_image_v2(image_path):
    """
    COMPLETE CLEAN VERSION: Extract 4 columns without duplicates and noise
    """
    print("DEBUG: Starting COMPLETE CLEAN extraction")

    try:
        # Enhanced preprocessing
        processed = simple_preprocess(image_path)
        temp_path = save_temp_image(processed)

        # Multiple OCR approaches
        all_ocr_items = []

        # Approach 1: Ultra-aggressive for handwritten
        try:
            ocr1 = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu=False,
                show_log=False,
                det_db_thresh=0.01,
                det_db_box_thresh=0.03,
                det_db_unclip_ratio=3.5,
                rec_batch_num=20
            )
            result1 = ocr1.ocr(temp_path, cls=True)
            if result1 and result1[0]:
                all_ocr_items.extend(result1[0])
                print(f"DEBUG: Ultra-aggressive OCR found {len(result1[0])} items")
        except Exception as e:
            print(f"DEBUG: Ultra-aggressive OCR failed: {e}")

        # Approach 2: Standard OCR
        try:
            ocr2 = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False,
                det_db_thresh=0.25,
                det_db_box_thresh=0.4,
                det_db_unclip_ratio=1.5
            )
            result2 = ocr2.ocr(image_path, cls=True)
            if result2 and result2[0]:
                # Add unique items based on text similarity
                existing_texts = {item[1][0].strip().lower() for item in all_ocr_items if len(item) >= 2}
                for item in result2[0]:
                    if len(item) >= 2:
                        text = item[1][0].strip().lower()
                        if text not in existing_texts:
                            all_ocr_items.append(item)
                            existing_texts.add(text)

                print(f"DEBUG: Total after standard merge: {len(all_ocr_items)}")
        except Exception as e:
            print(f"DEBUG: Standard OCR failed: {e}")

        # Approach 3: Extremely aggressive bottom area processing for handwritten
        try:
            img = cv2.imread(image_path)
            if img is not None:
                height = img.shape[0]
                # Process bottom 30% where handwritten entries are located
                bottom_crop = img[int(height * 0.7):, :]

                if bottom_crop.size > 0:
                    # Apply additional preprocessing for handwritten text
                    bottom_gray = cv2.cvtColor(bottom_crop, cv2.COLOR_BGR2GRAY)
                    bottom_enhanced = cv2.bilateralFilter(bottom_gray, 9, 75, 75)

                    # Save enhanced bottom crop
                    temp_bottom = save_temp_image(bottom_enhanced)

                    # Extremely aggressive OCR for handwritten
                    ocr3 = PaddleOCR(
                        use_angle_cls=False,
                        lang='en',
                        use_gpu=False,
                        show_log=False,
                        det_db_thresh=0.001,  # Ultra-extreme threshold
                        det_db_box_thresh=0.005,
                        det_db_unclip_ratio=5.0,  # Maximum expansion
                        rec_batch_num=30
                    )
                    result3 = ocr3.ocr(temp_bottom, cls=True)

                    if result3 and result3[0]:
                        # Adjust coordinates back to full image
                        for item in result3[0]:
                            if len(item) >= 2:
                                poly = item[0]
                                for point in poly:
                                    point[1] += int(height * 0.7)

                        all_ocr_items.extend(result3[0])
                        print(f"DEBUG: Ultra-aggressive bottom processing found {len(result3[0])} items")

                    os.unlink(temp_bottom)

                # ADDITIONAL: Try the very bottom 15% with different preprocessing
                bottom_5_crop = img[int(height * 0.95):, :]

                if bottom_5_crop.size > 0:
                    # Enhance contrast and sharpen
                    bottom_5_gray = cv2.cvtColor(bottom_5_crop, cv2.COLOR_BGR2GRAY)
                    # Adaptive thresholding to pull out faint handwriting
                    thresh = cv2.adaptiveThreshold(bottom_5_gray, 255,
                                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, 15, 10)

                    kernel = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]])
                    sharpened = cv2.filter2D(thresh, -1, kernel)

                    temp_bottom_5 = save_temp_image(sharpened)

                    ocr5 = PaddleOCR(
                        use_angle_cls=False,
                        lang='en',
                        use_gpu=False,
                        show_log=False,
                        det_db_thresh=0.0002,
                        det_db_box_thresh=0.0005,
                        det_db_unclip_ratio=6.0,
                    )
                    result5 = ocr5.ocr(temp_bottom_5, cls=True)

                    if result5 and result5[0]:
                        for item in result5[0]:
                            if len(item) >= 2:
                                for point in item[0]:
                                    point[1] += int(height * 0.95)  # re-map Y
                        all_ocr_items.extend(result5[0])
                        print(f"DEBUG: Bottom 5% processing found {len(result5[0])} additional items")

                    os.unlink(temp_bottom_5)


        except Exception as e:
            print(f"DEBUG: Bottom processing failed: {e}")

        os.unlink(temp_path)

    except Exception as e:
        print(f"DEBUG: Fallback OCR: {e}")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr.ocr(image_path, cls=True)
        all_ocr_items = result[0] if result and result[0] else []

    if not all_ocr_items:
        raise RuntimeError("No OCR results")

    # Parse OCR items with very lenient filtering
    ocr_items = []
    for line in all_ocr_items:
        if line and len(line) >= 2:
            poly, (text, score) = line[0], line[1]

            # Very lenient confidence threshold
            if score < 0.05:
                continue

            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)

            # Minimal size filtering
            if width < 2 or height < 2:
                continue

            ocr_items.append({
                'text': text.strip(),
                'score': float(score),
                'x': sum(xs) / len(xs),
                'y': sum(ys) / len(ys),
                'width': width,
                'height': height
            })

    print(f"DEBUG: Parsed {len(ocr_items)} OCR items")

    # Sort by Y coordinate
    ocr_items.sort(key=lambda x: x['y'])

    def is_person_name(text):
        """Improved name detection to eliminate noise"""
        text = text.strip()

        if len(text) < 3:
            return False

        lower = text.lower()

        # Reject known column headers
        if lower in ['name', 'payment', 'toa', 'mode', 'signature']:
            return False

        # Reject numeric or time-like values
        if re.match(r'^\d+$', lower) or re.match(r'^\d{1,2}[:.]\d{1,2}$', lower):
            return False

        # Reject very short single-word
        if len(text) <= 4 and ' ' not in text:
            return False

        # Reject known invalid patterns (expanded list)
        noise = [
            'cash', 'done', 'online', 'cosh', 'cagh', 'upi',
            'slla', 'caoh', 'cosh', 'fulatomd', 'wine', 'pony', 'ak17', 'rgodti',
            'cast', 'cagu', 'cogn', 'sub', 'ooo', 'owgou', 'pepol','cash.'
        ]
        if lower in noise:
            return False

        # Reject words with all uppercase (often false positives like 'FULATOMD')
        if text.isupper() and len(text) <= 10:
            return False

        # Must contain at least 4 letters
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars < 4:
            return False

        # Must contain mostly alphabetic characters
        total_chars = len(text.replace(' ', '').replace('.', ''))
        if total_chars == 0 or alpha_chars / total_chars < 0.8:
            return False

        # Looks like a name: capitalized words
        words = text.split()
        if all(w[0].isupper() for w in words if w):
            return True

        return False

    def is_payment_amount(text):
        """Enhanced payment detection"""
        text = text.strip()

        # Clean the text
        clean = text.replace('₹', '').replace('Rs', '').replace(',', '').replace(' ', '')

        # Direct numeric check
        if clean.isdigit():
            amount = int(clean)
            return 50 <= amount <= 99999

        # Handle OCR errors in payments
        numeric_match = re.search(r'\d{3,5}', clean)
        if numeric_match:
            amount = int(numeric_match.group())
            return 100 <= amount <= 50000

        return False

    def is_time_format(text):
        """Detect time values"""
        patterns = [
            r'^\d{1,2}[:.]\d{1,2}$',
            r'^\d{1,2}[:.]\d{1,2}\s*[AP]M?$'
        ]
        return any(re.match(pattern, text.strip()) for pattern in patterns)

    def is_mode_value(text):
        """Detect payment mode"""
        modes = ['cash', 'online', 'done', 'cosh', 'cagh']
        return text.lower().strip() in modes

    # Find table start
    table_start_y = 0
    for item in ocr_items:
        if 'name' in item['text'].lower() and len(item['text']) < 10:
            table_start_y = item['y'] + 15
            print(f"DEBUG: Table starts at y={table_start_y}")
            break

    # Filter to table area
    table_items = [item for item in ocr_items if item['y'] >= table_start_y]
    print(f"DEBUG: {len(table_items)} items in table area")

    # Get all unique person names with special handling for handwritten
    all_names = []

    # First pass: collect obvious printed names
    for item in table_items:
        if is_person_name(item['text']):
            all_names.append(item)

    # Second pass: be more lenient for bottom area (handwritten)
    max_y = max(item['y'] for item in table_items) if table_items else 0
    bottom_threshold = max_y - 100  # Bottom 150 pixels

    print(f"DEBUG: Looking for handwritten names in bottom area (y >= {bottom_threshold})")

    for item in table_items:
        if item['y'] >= bottom_threshold:
            text = item['text'].strip()

            if (len(text) >= 3 and
                    text[0].isalpha() and
                    sum(c.isalpha() for c in text) >= 3 and
                    not any(keyword in text.lower() for keyword in
                            ['cash', 'done', 'online', 'payment', 'signature', 'mode', 'toa']) and
                    not re.match(r'^\d{1,4}[:.]\d{2}$', text) and
                    not re.match(r'^\d{3,6}$', text)):

                # If not already in all_names
                if text.lower() not in [n['text'].lower() for n in all_names]:
                    all_names.append(item)
                    print(f"DEBUG: Added handwritten name: '{text}' at y={item['y']:.0f}")

                # Check if it's a potential name we haven't seen
                is_duplicate = False
                for existing in all_names:
                    if (abs(item['y'] - existing['y']) < 15 or  # Very close Y
                            text.lower() in existing['text'].lower() or
                            existing['text'].lower() in text.lower()):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    all_names.append(item)
                    print(f"DEBUG: Added potential handwritten name: '{text}' at y={item['y']:.0f}")

    # Remove duplicates and sort by Y coordinate
    from difflib import SequenceMatcher

    def name_similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # Improved duplicate filtering
    unique_names = []
    for item in all_names:
        is_duplicate = False
        for existing in unique_names:
            y_close = abs(item['y'] - existing['y']) < 12
            similarity = name_similarity(item['text'], existing['text'])

            if y_close and similarity > 0.85:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_names.append(item)

    # Update all_names to deduplicated and sorted
    all_names = sorted(unique_names, key=lambda x: x['y'])

    print(f"DEBUG: Found {len(all_names)} unique person names:")
    for i, name_item in enumerate(all_names):
        print(f"  {i + 1:2d}: '{name_item['text']}' at y={name_item['y']:.0f}")

    # For each name, find associated data in same row
    records = []

    for name_item in all_names:
        name_y = name_item['y']
        name_x = name_item['x']
        name_text = name_item['text']

        # Find items in same row (within 25 pixels vertically)
        row_items = []
        for item in table_items:
            if (abs(item['y'] - name_y) <= 25 and
                    item['x'] > name_x and  # To the right of name
                    item['text'] != name_text):  # Not the name itself
                row_items.append(item)

        # Create record
        record = {
            'name': name_text,
            'payment': '',
            'toa': '',
            'mode': ''
        }

        # Assign items to columns based on content type
        for item in row_items:
            text = item['text'].strip()

            if is_time_format(text) and not record['toa']:
                record['toa'] = text
            elif is_payment_amount(text) and not record['payment']:
                record['payment'] = text
            elif is_mode_value(text) and not record['mode']:
                record['mode'] = text

        records.append(record)

    print(f"DEBUG: Created {len(records)} individual records")

    # Show final records
    print("\nDEBUG: Final extracted records:")
    for i, record in enumerate(records):
        print(
            f"  {i + 1:2d}: '{record['name'][:30]:30s}' | Pay: '{record['payment']:8s}' | TOA: '{record['toa']:8s}' | Mode: '{record['mode']}'")

    # Convert to DataFrame
    if records:
        df_data = []
        for record in records:
            df_data.append([
                record['name'],
                record['payment'],
                record['toa'],
                record['mode']
            ])

        df = pd.DataFrame(df_data, columns=COLUMNS)
    else:
        df = pd.DataFrame([['No valid data found', '', '', '']], columns=COLUMNS)

    # Remove any completely empty rows
    df = df[df.apply(lambda x: any(x.astype(str).str.strip() != ''), axis=1)]

    print(f"DEBUG: Final DataFrame shape: {df.shape}")
    return df

def test_complete_extraction(image_path):
    """Test complete extraction without duplicates and noise"""
    try:
        print("=" * 60)
        print("COMPLETE CLEAN EXTRACTION - NO NOISE, INCLUDES HANDWRITTEN")
        print("=" * 60)

        result_df = extract_data_from_image_v2(image_path)

        print(f"\nFinal Results - {len(result_df)} clean records:")
        print(result_df.to_string(index=False, max_colwidth=30))

        return result_df

    except Exception as e:
        print(f"Complete extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    image_path = "accounting.jpeg"

    if os.path.exists(image_path):
        df = test_complete_extraction(image_path)
        if df is not None:
            print(f"\nSuccess! Extracted {len(df)} clean records with no noise.")

            # Save to CSV
            output_path = "extracted_complete_clean.csv"
            df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        else:
            print("Extraction failed.")
    else:
        print(f"Image file not found: {image_path}")