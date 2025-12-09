from nltk.corpus import wordnet as wn

pos = wn.ADJ
offset = 635456

synset = wn.synset_from_pos_and_offset(pos, offset)

print(f"Synset: {synset}")
print(f"Definition: {synset.definition()}")
print(f"Lemmas: {synset.lemma_names()}")
