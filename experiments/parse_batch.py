import re
import sys

SPECIAL_KEYS = [
    ('NAME', True),
    ('SCRIPT', True),
]

def is_special_key(key):
    for k, _ in SPECIAL_KEYS:
        if key == k:
            return True
    return False

class ParseError(Exception):

    def __init__(self, error, block=None):
        super().__init__()
        self.error = error
        self.block = block

    def message(self):
        if self.block is None:
            return self.error
        else:
            return f'{self.error}:\n{self.block}'


def is_alias(alias):
    alias_regex = '[a-zA-Z0-9]*'
    return bool(re.fullmatch(alias_regex, alias))


class Value:

    def __init__(self, value, alias, display=True):
        self.value = value
        if alias is None:
            self.alias = value
        else:
            self.alias = alias
        if display and not is_alias(self.alias):
            raise ParseError(f'Illegal alias {self.alias} for value {self.value}')


class Key:

    def __init__(self, name, alias, display):
        self.name = name
        if alias is None:
            self.alias = name 
        else:
            self.alias = alias
        if display and not is_alias(self.alias):
            raise ParseError(f'Illegal alias {self.alias} for key {self.name}')
        self.display = display
        self.values = []

    def add_value(self, value):
        if value in self.values:
            sys.stderr.write('WARNING: duplicate value {value} for key {self.name}')
        self.values.append(value)


def strip_comments(s):
    return '\n'.join([line.split('#')[0] for line in s.split('\n')])

def join_and_strip(s):
    return s.replace('\n', ' ').replace('\t', ' ').strip()


def parse_optional_alias(alias_text):
    alias_text = alias_text.strip()
    if not alias_text:
        return None

    alias_regex = '\\( *([a-zA-Z0-9]*) *\\)'
    match = re.fullmatch(alias_regex, alias_text)
    if not match:
        raise ParseError('Expected an <optional-alias>', alias_text)
    return match.group(1)

def parse_key(key_text):
    key_text = join_and_strip(key_text)
    parts = key_text.split()
    if len(parts) == 0:
        raise ParseError('Missing <key>')

    no_display = False
    key_name = parts[0]
    if key_name.endswith('?'):
        no_display = True
        key_name = key_name[:-1]
    if is_special_key(key_name):
        if no_display:
            raise ParseError(f'Special key {key_name} should not be marked as no-display')
        no_display = True

    key_regex = '[a-zA-Z_][a-zA-Z0-9_]*'
    if not re.fullmatch(key_regex, key_name):
        raise ParseError(f'Invalid key name {key_name}', key_text)

    alias = parse_optional_alias(' '.join(parts[1:]))
    
    return Key(key_name, alias, not no_display)


def parse_key_tuple(key_tuple_text, key_names):
    key_tuple = []
    keys_text = key_tuple_text.split(',')
    if not keys_text:
        raise ParseError('Expecting one or more keys', key_tuple_text)
    for key_text in keys_text:
        key = parse_key(key_text)
        if key.name in key_names:
            raise ParseError(f'Duplicate key name {key.name}')
        key_names.add(key.name)
        key_tuple.append(key)
    return key_tuple

def parse_value(value_text, display):
    value_text = join_and_strip(value_text)
    parts = value_text.split()
    if len(parts) == 0:
        raise ParseError('Missing <value>')

    value = parts[0]
    alias = parse_optional_alias(' '.join(parts[1:]))
    
    return Value(value, alias, display=display)

def parse_value_row(row_text, key_tuple):
    values_text = row_text.split(',')
    if len(values_text) != len(key_tuple):
        key_names = ', '.join([k.name for k in key_tuple])
        raise ParseError(
            f'Expecting {len(key_tuple)} values to match keys {key_names}',
            row_text
        )

    for key, value_text in zip(key_tuple, values_text):
        key.add_value(parse_value(value_text, key.display))

def parse_csv_values(csv_path, key_tuple):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ParseError(str(e))
    for key in key_tuple:
        if key.name not in df:
            raise ParseError(f'CSV file {csv_path} does not contain column {key.name}')
        for value in df[key.name]:
            key.add_value(parse_value(str(value), key.display))

def parse_value_tuple(value_tuple_text, key_tuple):
    value_tuple_text = value_tuple_text.strip()
    if not value_tuple_text:
        raise ParseError('Expected <value-tuple>')

    rows_text = value_tuple_text.split('|')
    if not rows_text:
        raise ParseError('Expecting one or more values', value_tuple_text)

    for row_text in rows_text:
        parse_value_row(row_text, key_tuple)

    rows = set()
    for i in range(len(key_tuple[0].values)):
        row = tuple(k.values[i] for k in key_tuple)
        if row in rows:
            key_names = ', '.join([k.name for k in key_tuple])
            raise ParseError(
                f'Duplicate value-row {row} for key-tuple {key_names}',
                value_tuple_text
            )
        rows.add(row)


def parse_kvp(kvp, key_names):
    parts = kvp.split('=')
    if len(parts) != 2:
        raise ParseError('Expected <key-tuple>=<value-tuple>', kvp)
    key_tuple = parse_key_tuple(parts[0], key_names)
    parse_value_tuple(parts[1], key_tuple)
    return key_tuple


def get_special_value(key, key_tuples, required=True):
    for key_tuple in key_tuples:
        if key in [k.name for k in key_tuple]:
            if len(key_tuple) != 1:
                raise ParseError(
                    f'Special key {key} may not share a tuple with another key'
                )
            if len(key_tuple[0].values) != 1:
                raise ParseError(
                    f'Special key {key} may only have one value'
                )
            return key_tuple[0].values[0].value

    if required:
        raise ParseError(f'Must supply a value for special key {key}')
    else:
        return '-'


def enumerate_jobs(key_tuples, i, name, kvps, first_display=True):
    if i == len(key_tuples):
        kpv_str = ' '.join([
            f'{k}={v}' for k, v in kvps
        ])
        print(f'{name} {kpv_str}')
        return

    key_tuple = key_tuples[i]

    if len(key_tuple[0].values) > 1:
        represented = False
        for k in key_tuple:
            if k.display:
                represented = True
        if not represented:
            key_names = ', '.join([k.name for k in key_tuple])
            raise ParseError(f'At least one of the keys {key_names} must be displayed')

    for j in range(len(key_tuple[0].values)):
        augmented_kvps = kvps[:]
        augmented_name = name
        still_first_display = first_display
        for k in key_tuple:
            augmented_kvps.append((k.name, k.values[j].value))
            if k.display:
                key_str = f'{k.alias}-' if k.alias else ''
                value_str = k.values[j].alias
                joiner = '/' if still_first_display else '_'
                still_first_display = False
                augmented_name = f'{augmented_name}{joiner}{key_str}{value_str}'

        enumerate_jobs(key_tuples, i + 1, augmented_name, augmented_kvps, first_display=still_first_display)


def main():
    key_names = set()
    key_tuples = []

    text = sys.stdin.read()
    text = strip_comments(text)
    
    kvps = text.split(';')
    for kvp in kvps:
        kvp = kvp.strip()
        if kvp:
            key_tuples.append(parse_kvp(kvp, key_names))

    special_kvps = {}
    for key, required in SPECIAL_KEYS:
        special_kvps[key] = get_special_value(key, key_tuples, required=required)

    batch_description = ' '.join([special_kvps[k] for k, _ in SPECIAL_KEYS])
    print(batch_description)

    key_tuples = [k for k in key_tuples if len(k) > 1 or k[0].name not in special_kvps]
    
    n_jobs = 1
    for key_tuple in key_tuples:
        n_jobs *= len(key_tuple[0].values)
    if n_jobs > 1000:
        raise ParseError(f'Attempting to create too many jobs: {n_jobs}')
    
    enumerate_jobs(key_tuples, 0, special_kvps['NAME'], [])


if __name__ == '__main__':
    try:
        main()
    except ParseError as e:
        sys.stderr.write(e.message() + '\n')
        sys.exit(1)

