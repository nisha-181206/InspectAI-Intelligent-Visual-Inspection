def calculate_zone(bbox, image_width, image_height):
    """
    Determine the location of a defect using a 3x3 grid.

    Returns:
        zone_name
        zone_row
        zone_column
        center_x
        center_y
        relative_x
        relative_y
    """

    x1, y1, x2, y2 = bbox

    # --------------------------------------------------
    # Calculate defect center
    # --------------------------------------------------

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # --------------------------------------------------
    # Relative position
    # --------------------------------------------------

    relative_x = center_x / image_width
    relative_y = center_y / image_height

    # --------------------------------------------------
    # Determine grid column
    # --------------------------------------------------

    if relative_x < 1 / 3:
        column = 0
    elif relative_x < 2 / 3:
        column = 1
    else:
        column = 2

    # --------------------------------------------------
    # Determine grid row
    # --------------------------------------------------

    if relative_y < 1 / 3:
        row = 0
    elif relative_y < 2 / 3:
        row = 1
    else:
        row = 2

    # --------------------------------------------------
    # Zone names
    # --------------------------------------------------

    zone_names = [
        ["Top-Left", "Top-Center", "Top-Right"],
        ["Middle-Left", "Center", "Middle-Right"],
        ["Bottom-Left", "Bottom-Center", "Bottom-Right"]
    ]

    zone_name = zone_names[row][column]

    return {
        "zone": zone_name,
        "zone_row": row + 1,
        "zone_column": column + 1,
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "relative_x": round(relative_x, 3),
        "relative_y": round(relative_y, 3)
    }