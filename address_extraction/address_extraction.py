try:
    from address_extraction.address_extractor import AddressExtractor
except ImportError:
    from address_extractor import AddressExtractor

MAX_SCALE = 3.01
SCALE_STEP = 0.5

ae = AddressExtractor()

def extract_addresses(fields_address, max_scale=MAX_SCALE, scale_step=SCALE_STEP):
    return ae.run(fields_address, MAX_SCALE, SCALE_STEP)