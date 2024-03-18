def difference(input_data):
    # Define persistent variable
    if not hasattr(difference, 'temp_input'):
        difference.temp_input = input_data
    
    # Calculate the difference
    output = input_data - difference.temp_input
    
    # Update the persistent variable
    difference.temp_input = input_data
    
    return output