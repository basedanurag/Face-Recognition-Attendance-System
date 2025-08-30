print("🔧 FIXING THE RECOGNITION MAPPING ISSUE...")

# Read the file
with open("main_final_enhanced_fixed.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the broken recognition logic
old_logic = """                            # Debug: print what we are looking for
                            print(f"Looking for SERIAL NO. == {serial}")
                            
                            # Find by serial number
                            matching_rows = df[df["SERIAL NO."] == serial]
                            print(f"Matching rows: {len(matching_rows)}")
                            
                            if len(matching_rows) > 0:
                                name_col = matching_rows["NAME"].iloc[0]
                                id_col = matching_rows["ID"].iloc[0]"""

new_logic = """                            # Debug: print what we are looking for  
                            print(f"Recognition returned ID: {serial}, looking for this ID in database")
                            
                            # Find by ID number (serial is actually the ID from training)
                            matching_rows = df[df["ID"] == serial]
                            print(f"Matching rows for ID {serial}: {len(matching_rows)}")
                            
                            if len(matching_rows) > 0:
                                name_col = matching_rows["NAME"].iloc[0]
                                id_col = matching_rows["ID"].iloc[0]
                                serial_col = matching_rows["SERIAL NO."].iloc[0]"""

content = content.replace(old_logic, new_logic)

# Write the fixed version
with open("main_final_enhanced_fixed.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✓ Fixed the recognition mapping logic!")
print("✓ Now it will correctly map recognized IDs to database records")
