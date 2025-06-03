import json
import pandas as pd

path = 'datasets/classified_images_final.json'

with open(path) as f:
    data = json.load(f)

fish_taxonomy = {
    'kingdom': {}
}

fish_path = "datasets/fish_data/final.csv"
fish_data = pd.read_csv(fish_path)
for _, row in fish_data.iterrows():
    kingdom = "1"
    phylum = "1"
    class_ = row['Class']
    order = row['Order'] 
    family = row['Family']
    genus = row['Genus']
    species = row['species'] if not pd.isna(row['species']) else 'any'

    if kingdom not in fish_taxonomy['kingdom']:
        fish_taxonomy['kingdom'][kingdom] = {}
    
    if phylum not in fish_taxonomy['kingdom'][kingdom]:
        fish_taxonomy['kingdom'][kingdom][phylum] = {}
    
    if class_ not in fish_taxonomy['kingdom'][kingdom][phylum]:
        fish_taxonomy['kingdom'][kingdom][phylum][class_] = {}
    
    if order not in fish_taxonomy['kingdom'][kingdom][phylum][class_]:
        fish_taxonomy['kingdom'][kingdom][phylum][class_][order] = {}
    
    if family not in fish_taxonomy['kingdom'][kingdom][phylum][class_][order]:
        fish_taxonomy['kingdom'][kingdom][phylum][class_][order][family] = {}
    
    if genus not in fish_taxonomy['kingdom'][kingdom][phylum][class_][order][family]:
        fish_taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus] = []
    
    fish_taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus].append(species)


bio_scan_taxonomy = {
    'kingdom': {}
}

bio_scan_path = "datasets/bio_scan_data/data.csv"
bio_scan_data = pd.read_csv(bio_scan_path)
for _, row in bio_scan_data.iterrows():
    kingdom = "1"
    phylum = "1"
    class_ = row['class']
    order = row['order'] 
    family = row['family']
    genus = row['genus']
    species = row['species']

    if kingdom not in bio_scan_taxonomy['kingdom']:
        bio_scan_taxonomy['kingdom'][kingdom] = {}
    
    if phylum not in bio_scan_taxonomy['kingdom'][kingdom]:
        bio_scan_taxonomy['kingdom'][kingdom][phylum] = {}
    
    if class_ not in bio_scan_taxonomy['kingdom'][kingdom][phylum]:
        bio_scan_taxonomy['kingdom'][kingdom][phylum][class_] = {}
    
    if order not in bio_scan_taxonomy['kingdom'][kingdom][phylum][class_]:
        bio_scan_taxonomy['kingdom'][kingdom][phylum][class_][order] = {}
    
    if family not in bio_scan_taxonomy['kingdom'][kingdom][phylum][class_][order]:
        bio_scan_taxonomy['kingdom'][kingdom][phylum][class_][order][family] = {}
    
    if genus not in bio_scan_taxonomy['kingdom'][kingdom][phylum][class_][order][family]:
        bio_scan_taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus] = []
    
    bio_scan_taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus].append(species)

# Initialize taxonomy mappings
taxonomy = {
    'kingdom': {}
}

# Build the taxonomy hierarchy from the dataset
for category in data['categories']:
    kingdom = category['kingdom']
    phylum = category['phylum']
    class_ = category['class']
    order = category['order']
    family = category['family']
    genus = category['genus']
    species = category['specific_epithet']
    
    if kingdom not in taxonomy['kingdom']:
        taxonomy['kingdom'][kingdom] = {}
    
    if phylum not in taxonomy['kingdom'][kingdom]:
        taxonomy['kingdom'][kingdom][phylum] = {}
    
    if class_ not in taxonomy['kingdom'][kingdom][phylum]:
        taxonomy['kingdom'][kingdom][phylum][class_] = {}
    
    if order not in taxonomy['kingdom'][kingdom][phylum][class_]:
        taxonomy['kingdom'][kingdom][phylum][class_][order] = {}
    
    if family not in taxonomy['kingdom'][kingdom][phylum][class_][order]:
        taxonomy['kingdom'][kingdom][phylum][class_][order][family] = {}
    
    if genus not in taxonomy['kingdom'][kingdom][phylum][class_][order][family]:
        taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus] = []
    
    taxonomy['kingdom'][kingdom][phylum][class_][order][family][genus].append(species)


def get_keys_at_level(taxonomy, level):
    """
    Given a taxonomic level, return all distinct keys (names) at that level.
    
    Args:
    taxonomy (dict): The full taxonomy dictionary.
    level (str): The taxonomic level to search for (e.g., 'kingdom', 'phylum', 'class', 'order', 'family', 'genus').

    Returns:
    list: A list of unique names at the specified level.
    """
    keys_list = set()  # Use a set to avoid duplicates

    # Traverse the taxonomy tree and collect distinct keys at the given level
    if level == 'kingdom':
        keys_list.update(taxonomy['kingdom'].keys())
    
    elif level == 'phylum':
        for kingdom, phyla in taxonomy['kingdom'].items():
            keys_list.update(phyla.keys())
    
    elif level == 'class':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                keys_list.update(classes.keys())
    
    elif level == 'order':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    keys_list.update(orders.keys())
    
    elif level == 'family':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        keys_list.update(families.keys())
    
    elif level == 'genus':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        for family, genera in families.items():
                            keys_list.update(genera.keys())

    elif level == 'specific_epithet' or level == 'species':  # This is the species level
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        for family, genera in families.items():
                            for genus, species in genera.items():
                                keys_list.update(species)

    keys_list = {item for item in keys_list if isinstance(item, str)}

    print(keys_list)

    return sorted(keys_list)


def get_lineage_bottom_to_top(taxonomy, target_name, level, logging=None):
    # Recursive function to traverse bottom-up from any level to kingdom
    def traverse(kingdom, phylum, class_, order, family, genus, species):
        if species:
            return {
                'species': species,
                'genus': genus,
                'family': family,
                'order': order,
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if genus:
            return {
                'genus': genus,
                'family': family,
                'order': order,
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if family:
            return {
                'family': family,
                'order': order,
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if order:
            return {
                'order': order,
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if class_:
            return {
                'class': class_,
                'phylum': phylum,
                'kingdom': kingdom
            }
        if phylum:
            return {
                'phylum': phylum,
                'kingdom': kingdom
            }
        if kingdom:
            return {
                'kingdom': kingdom
            }

    # Searching based on level
    if level == 'species':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        for family, genera in families.items():
                            for genus, species_list in genera.items():
                                if target_name in species_list:
                                    return traverse(kingdom, phylum, class_, order_, family, genus, target_name)
    
    elif level == 'genus':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        for family, genera in families.items():
                            if target_name in genera:
                                return traverse(kingdom, phylum, class_, order_, family, target_name, None)
    
    elif level == 'family':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    for order_, families in orders.items():
                        if target_name in families:
                            return traverse(kingdom, phylum, class_, order_, target_name, None, None)
    
    elif level == 'order':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                for class_, orders in classes.items():
                    if target_name in orders:
                        return traverse(kingdom, phylum, class_, target_name, None, None, None)
    
    elif level == 'class':
        for kingdom, phyla in taxonomy['kingdom'].items():
            for phylum, classes in phyla.items():
                if target_name in classes:
                    return traverse(kingdom, phylum, target_name, None, None, None, None)
    
    elif level == 'phylum':
        for kingdom, phyla in taxonomy['kingdom'].items():
            if target_name in phyla:
                return traverse(kingdom, target_name, None, None, None, None, None)
    
    elif level == 'kingdom':
        if target_name in taxonomy['kingdom']:
            return traverse(target_name, None, None, None, None, None, None)
    
    return None


def load_mappings(mapping_file="condition_mappings.txt"):
    mappings = {
        'kingdom': {}, 'phylum': {}, 'class': {}, 
        'order': {}, 'family': {}, 'genus': {}, 
        'specific_epithet': {}
    }
    with open(mapping_file, 'r') as f:
        current_key = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in mappings:
                current_key = line
            elif current_key:
                value, index = line.split(': ')
                mappings[current_key][value] = int(index)
    return mappings


def map_condition(category, condition_mappings):
    """Map condition values to integer indices with cascading None behavior."""
    conditions = {}
    conditions_list = []
    for key in condition_mappings.keys():
        if key in category:
            value = category[key]
            conditions[key] = condition_mappings[key].get(value, condition_mappings[key]['None'])
            conditions_list.append(conditions[key])
        else:
            conditions[key] = condition_mappings[key]['None']
            conditions_list.append(conditions[key])
    return conditions, conditions_list


def get_paths_at_level(taxonomy, level):
    """
    Recursively traverse the taxonomy structure to collect paths up to a given taxonomic level.
    
    Args:
    taxonomy (dict): The full taxonomy dictionary.
    level (int): The taxonomic level to retrieve paths for (1=kingdom, 2=phylum, ..., 7=species).

    Returns:
    list: A list of dictionaries, each representing a full path up to the given level.
    """
    paths = []

    def traverse(node, current_path, current_level):
        # Base case: if we've reached the target level, append the current path
        if current_level == level:
            paths.append(current_path.copy())
            return
        
        # Recursive case: continue traversing the tree
        if current_level == 1:  # We are at the 'kingdom' level
            for kingdom, phyla in node['kingdom'].items():
                traverse(phyla, {**current_path, 'kingdom': kingdom}, current_level + 1)
        
        elif current_level == 2:  # We are at the 'phylum' level
            for phylum, classes in node.items():
                traverse(classes, {**current_path, 'phylum': phylum}, current_level + 1)
        
        elif current_level == 3:  # We are at the 'class' level
            for class_, orders in node.items():
                traverse(orders, {**current_path, 'class': class_}, current_level + 1)
        
        elif current_level == 4:  # We are at the 'order' level
            for order, families in node.items():
                traverse(families, {**current_path, 'order': order}, current_level + 1)
        
        elif current_level == 5:  # We are at the 'family' level
            for family, genera in node.items():
                traverse(genera, {**current_path, 'family': family}, current_level + 1)
        
        elif current_level == 6:  # We are at the 'genus' level
            for genus, species_list in node.items():
                traverse(species_list, {**current_path, 'genus': genus}, current_level + 1)
        
        elif current_level == 7:  # We are at the 'species' level
            for species in node:
                paths.append({**current_path, 'species': species})

    # Start the traversal from the root at level 1 (kingdom)
    traverse(taxonomy, {}, 1)

    return paths
