# ---------------------------------------------------------------------------
# REFERENCE ONLY -- not runnable from this repository.
#
# This script belongs to the upstream pipeline that produced the analysis
# dataset. It requires a Google Street View API key and/or the panorama store
# held on the lab server, neither of which is redistributable. It is included
# to document how the published features were derived, not to be re-executed.
#
# Set PERSONALITY_SVI_ROOT and SVI_IMAGE_DIR if you adapt it to your own data.
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

# Create a direct mapping between BFI and BFI-2 items where there's content overlap
def create_content_overlap_mapping():
    content_mapping = {
        # Extraversion mappings
        'ext01': ['bfi2_ext10'],  # Is talkative -> Is talkative
        'ext02r': ['bfi2_ext04r'],  # Is reserved -> Tends to be quiet
        'ext03': ['bfi2_ext09'],  # Is full of energy -> Is full of energy
        'ext04': ['bfi2_ext12'],  # Generates enthusiasm -> Shows enthusiasm
        'ext05r': ['bfi2_ext04r'],  # Tends to be quiet -> Tends to be quiet
        'ext06': ['bfi2_ext02'],  # Has assertive personality -> Has assertive personality
        'ext07r': ['bfi2_ext07r'],  # Is shy, inhibited -> Is shy, introverted
        'ext08': ['bfi2_ext01'],  # Is outgoing, sociable -> Is outgoing, sociable
        
        # Agreeableness mappings
        'agr01r': ['bfi2_agr03r'],  # Finds fault with others -> Finds fault with others
        'agr02': ['bfi2_agr07'],  # Is helpful and unselfish -> Is helpful and unselfish
        'agr03r': ['bfi2_agr05r'],  # Starts quarrels -> Starts arguments
        'agr04': ['bfi2_agr06'],  # Has forgiving nature -> Has forgiving nature
        'agr05': ['bfi2_agr12'],  # Is generally trusting -> Assumes the best about people
        'agr06r': ['bfi2_agr10r'],  # Can be cold and aloof -> Can be cold and uncaring
        'agr07': ['bfi2_agr01'],  # Is considerate and kind -> Is compassionate
        'agr08r': ['bfi2_agr08r'],  # Is sometimes rude -> Is sometimes rude
        'agr09': ['bfi2_agr11'],  # Likes to cooperate -> Is polite, courteous
        
        # Conscientiousness mappings
        'cns01': ['bfi2_cns08'],  # Does thorough job -> Is efficient, gets things done
        'cns02r': ['bfi2_cns06r'],  # Can be careless -> Can be somewhat careless
        'cns03': ['bfi2_cns09'],  # Is reliable worker -> Is reliable
        'cns04r': ['bfi2_cns01r'],  # Tends to be disorganized -> Tends to be disorganized
        'cns05r': ['bfi2_cns02r'],  # Tends to be lazy -> Tends to be lazy
        'cns06': ['bfi2_cns11'],  # Perseveres until task finished -> Is persistent until finished
        'cns07': ['bfi2_cns08'],  # Does things efficiently -> Is efficient
        'cns08': ['bfi2_cns03'],  # Makes plans and follows through -> Is dependable
        'cns09r': ['bfi2_cns05r'],  # Is easily distracted -> Has difficulty getting started
        
        # Neuroticism mappings
        'neu01': ['bfi2_neu11'],  # Is depressed, blue -> Tends to feel depressed, blue
        'neu02r': ['bfi2_neu01r'],  # Is relaxed, handles stress -> Is relaxed, handles stress
        'neu03': ['bfi2_neu04'],  # Can be tense -> Can be tense
        'neu04': ['bfi2_neu07'],  # Worries a lot -> Worries a lot
        'neu05r': ['bfi2_neu06r'],  # Is emotionally stable -> Is emotionally stable
        'neu06': ['bfi2_neu03'],  # Can be moody -> Is moody, has mood swings
        'neu07r': ['bfi2_neu09r'],  # Remains calm in tense situations -> Keeps emotions under control
        'neu08': ['bfi2_neu10r'],  # Gets nervous easily -> Rarely feels anxious (reversed mapping)
        
        # Openness mappings
        'opn01': ['bfi2_opn12'],  # Is original, has new ideas -> Is original, new ideas
        'opn02': ['bfi2_opn02'],  # Is curious about many things -> Is curious about many things
        'opn03': ['bfi2_opn08'],  # Is ingenious, deep thinker -> Is complex, deep thinker
        'opn04': ['bfi2_opn09r'],  # Has active imagination -> Has difficulty imagining (reversed mapping)
        'opn05': ['bfi2_opn03'],  # Is inventive -> Is inventive
        'opn06': ['bfi2_opn07'],  # Values artistic experiences -> Values art and beauty
        'opn07r': ['bfi2_opn05r'],  # Prefers routine work -> Avoids intellectual discussions (loose mapping)
        'opn08': ['bfi2_opn08'],  # Likes to reflect on ideas -> Is complex, deep thinker
        'opn09r': ['bfi2_opn01r'],  # Has few artistic interests -> Has few artistic interests
        'opn10': ['bfi2_opn04']  # Sophisticated in art/music -> Fascinated by art/music
    }
    
    return content_mapping

# Create a mapping with human-friendly names for the merged indicators
def create_human_readable_mapping():
    """
    Creates a mapping between BFI/BFI-2 items and human-readable names.
    Returns two dictionaries:
    1. A dictionary mapping the BFI item to a human-readable name
    2. A dictionary mapping the BFI-2 item to the same human-readable name
    
    All human-readable names are prefixed with 'ind_' for clarity and easier filtering.
    """
    # Define human-readable names for each content area
    human_readable = {
        # Extraversion
        'ext01': 'ind_talkative',
        'ext02r': 'ind_reserved_quiet',
        'ext03': 'ind_energetic',
        'ext04': 'ind_enthusiastic',
        'ext05r': 'ind_quiet_reserved',  # duplicate of ext02r, will handle specially
        'ext06': 'ind_assertive',
        'ext07r': 'ind_shy_inhibited',
        'ext08': 'ind_outgoing_sociable',
        
        # Agreeableness
        'agr01r': 'ind_critical_fault_finding',
        'agr02': 'ind_helpful_unselfish',
        'agr03r': 'ind_starts_quarrels_arguments',
        'agr04': 'ind_forgiving',
        'agr05': 'ind_trusting',
        'agr06r': 'ind_cold_aloof',
        'agr07': 'ind_considerate_kind',
        'agr08r': 'ind_sometimes_rude',
        'agr09': 'ind_cooperative_polite',
        
        # Conscientiousness
        'cns01': 'ind_thorough_efficient',
        'cns02r': 'ind_careless',
        'cns03': 'ind_reliable',
        'cns04r': 'ind_disorganized',
        'cns05r': 'ind_lazy',
        'cns06': 'ind_persistent_persevering',
        'cns07': 'ind_efficient',  # duplicate of cns01, will handle specially
        'cns08': 'ind_dependable_follows_through',
        'cns09r': 'ind_easily_distracted',
        
        # Neuroticism
        'neu01': 'ind_depressed_blue',
        'neu02r': 'ind_relaxed_handles_stress',
        'neu03': 'ind_tense',
        'neu04': 'ind_worries_a_lot',
        'neu05r': 'ind_emotionally_stable',
        'neu06': 'ind_moody',
        'neu07r': 'ind_calm_in_tense_situations',
        'neu08': 'ind_nervous_anxious',
        
        # Openness
        'opn01': 'ind_original_new_ideas',
        'opn02': 'ind_curious',
        'opn03': 'ind_deep_thinker',
        'opn04': 'ind_imagination',
        'opn05': 'ind_inventive',
        'opn06': 'ind_values_art_beauty',
        'opn07r': 'ind_prefers_routine_work',
        'opn08': 'ind_reflective',  # duplicate of opn03, will handle specially
        'opn09r': 'ind_few_artistic_interests',
        'opn10': 'ind_art_music_literature'
    }
    
    # Get the content mapping to link BFI and BFI-2 items
    content_mapping = create_content_overlap_mapping()
    
    # Create mappings for both BFI and BFI-2 items
    bfi_to_readable = {}
    bfi2_to_readable = {}
    
    # Fill the mappings
    for bfi_item, bfi2_items in content_mapping.items():
        readable_name = human_readable[bfi_item]
        
        # Handle duplicate readable names - append a number to distinguish
        if bfi_item == 'ext05r':  # This is a duplicate of ext02r
            readable_name = 'ind_reserved_quiet_2'
        elif bfi_item == 'cns07':  # This is a duplicate of cns01
            readable_name = 'ind_efficient_2'
        elif bfi_item == 'opn08':  # This is a duplicate of opn03
            readable_name = 'ind_deep_thinker_2'
            
        # Map the BFI item to the readable name
        bfi_to_readable[bfi_item] = readable_name
        
        # Map each corresponding BFI-2 item to the same readable name
        for bfi2_item in bfi2_items:
            bfi2_to_readable[bfi2_item] = readable_name
    
    return bfi_to_readable, bfi2_to_readable

def calculate_personality_scores_vectorized(df):
    """
    Calculate Big Five personality scores for all rows at once.
    This is a vectorized version that avoids row-by-row processing.
    
    This function handles both BFI and BFI-2 questionnaires:
    1. Calculates scores for each questionnaire separately
    2. Prioritizes BFI-2 when available, otherwise uses BFI
    3. Returns a dataframe with combined/merged scores
    4. Creates merged indicators with prefixed human-readable names (ind_*)
    """
    # Initialize results DataFrame
    results = pd.DataFrame(index=df.index)
    
    # Dictionary mapping traits to their prefixes
    trait_mapping = {
        'extraversion': 'ext',
        'agreeableness': 'agr',
        'conscientiousness': 'cns',
        'neuroticism': 'neu',
        'openness': 'opn'
    }
    
    # Function to identify and convert personality columns
    def get_personality_columns(prefix_set, exclude_prefixes=None):
        if exclude_prefixes is None:
            exclude_prefixes = []
            
        cols = []
        for col in df.columns:
            # Check if column starts with any of the target prefixes
            if any(col.startswith(prefix) for prefix in prefix_set):
                # Ensure it doesn't start with any excluded prefixes
                if not any(col.startswith(excl) for excl in exclude_prefixes):
                    cols.append(col)
        
        # Convert to numeric values
        if cols:
            return df[cols].apply(pd.to_numeric, errors='coerce')
        return None
    
    # Calculate BFI (original) scores
    bfi_prefixes = list(trait_mapping.values())
    # Exclude other-rated items (oext01, oagr01r, ocns01, oneu01, oopn01r, …) and
    # TIPI / BFI-2 columns.  Do NOT use the broad prefix 'o' here because
    # 'opn01'.startswith('o') is True and would accidentally drop ALL openness
    # items from the original BFI scoring (the bug this line fixes).
    bfi_scores = get_personality_columns(
        bfi_prefixes,
        exclude_prefixes=['oext', 'oagr', 'ocns', 'oneu', 'oopn', 'tipi', 'bfi2_'],
    )
    
    # Calculate BFI-2 scores
    bfi2_prefixes = [f'bfi2_{prefix}' for prefix in trait_mapping.values()]
    bfi2_scores = get_personality_columns(bfi2_prefixes)
    
    # Process each trait separately to calculate the main trait scores
    for trait, prefix in trait_mapping.items():
        # Calculate BFI trait score if available
        bfi_trait_score = None
        if bfi_scores is not None:
            trait_cols = [col for col in bfi_scores.columns if col.startswith(prefix)]
            if trait_cols:
                bfi_trait_score = bfi_scores[trait_cols].mean(axis=1)
        
        # Calculate BFI-2 trait score if available
        bfi2_trait_score = None
        if bfi2_scores is not None:
            bfi2_prefix = f'bfi2_{prefix}'
            trait_cols = [col for col in bfi2_scores.columns if col.startswith(bfi2_prefix)]
            if trait_cols:
                bfi2_trait_score = bfi2_scores[trait_cols].mean(axis=1)
        
        # Merge scores - prioritize BFI-2 if available, otherwise use BFI
        if bfi2_trait_score is not None and bfi_trait_score is not None:
            # Where BFI-2 is NaN, use BFI instead
            combined_score = bfi2_trait_score.combine_first(bfi_trait_score)
        elif bfi2_trait_score is not None:
            combined_score = bfi2_trait_score
        elif bfi_trait_score is not None:
            combined_score = bfi_trait_score
        else:
            combined_score = pd.Series(index=df.index, dtype=float)
        
        results[trait] = combined_score
    
    # Create merged indicators with prefixed human-readable names (ind_*)
    bfi_to_readable, bfi2_to_readable = create_human_readable_mapping()
    
    # Create the merged indicators
    for bfi_item, readable_name in bfi_to_readable.items():
        # For each human-readable indicator, try to get values from both BFI and BFI-2
        if bfi_scores is not None and bfi_item in bfi_scores.columns:
            bfi_values = bfi_scores[bfi_item]
        else:
            bfi_values = pd.Series(index=df.index, dtype=float)
        
        # Find the corresponding BFI-2 item(s)
        content_mapping = create_content_overlap_mapping()
        if bfi_item in content_mapping:
            bfi2_items = content_mapping[bfi_item]
            bfi2_values = None
            
            # Try to get values from each corresponding BFI-2 item
            for bfi2_item in bfi2_items:
                if bfi2_scores is not None and bfi2_item in bfi2_scores.columns:
                    if bfi2_values is None:
                        bfi2_values = bfi2_scores[bfi2_item]
                    else:
                        # If multiple BFI-2 items map to this BFI item, take their average
                        bfi2_values = (bfi2_values + bfi2_scores[bfi2_item]) / 2
            
            # Merge the values, prioritizing BFI-2
            if bfi2_values is not None:
                combined_values = bfi2_values.combine_first(bfi_values)
            else:
                combined_values = bfi_values
        else:
            combined_values = bfi_values
        
        # Add the merged indicator to the results
        if not combined_values.isna().all():  # Only add if there's at least one non-NA value
            results[readable_name] = combined_values
    
    # Report availability of each questionnaire
    bfi_count = bfi_scores.notna().any(axis=1).sum() if bfi_scores is not None else 0
    bfi2_count = bfi2_scores.notna().any(axis=1).sum() if bfi2_scores is not None else 0
    ind_count = len([col for col in results.columns if col.startswith('ind_')])

    print(f"  Using BFI data for {bfi_count} rows and BFI-2 data for {bfi2_count} rows")
    print(f"  Calculated {len(trait_mapping)} personality traits and {ind_count} human-readable indicators")

    # only keep the trait and human-readable columns
    results = results[[trait for trait in trait_mapping.keys()] + [col for col in results.columns if col.startswith('ind_')]]
    
    return results
