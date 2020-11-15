from language_model import normalize_text
from ex2 import Spell_Checker
from spelling_confusion_matrices import error_tables

# corpora = open('corpora/big.txt').read()
corpora = 'A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat on the rat that sat on the cat on the mat. '
alphabet = 'abcdefghijklmnopqrstuvwxyz'
for i in alphabet:
    for j in alphabet:
        corpora += i + j
alpha = 0.95
text = 'a dat sat on the mat.'

spell_checker = Spell_Checker()
lm = spell_checker.build_model(corpora)
spell_checker.add_language_model(lm)
spell_checker.add_error_tables(error_tables)

print('original  : %s' % text)
print('corrected : %s' % spell_checker.spell_check(text, alpha))
