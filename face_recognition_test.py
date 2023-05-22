from face_recognition import load_image_file,face_encodings,compare_faces
known_image = load_image_file("image/obama.jpg")
unknown_image = load_image_file("image/unknown.jpg")
unknown_2 = load_image_file("image/unknown_2.jpg")
biden = load_image_file("image/biden.jpg")
biden_unknown = load_image_file("image/biden_unknown.jpg")

obama_encoding = face_encodings(known_image)[0]
unknown_encoding = face_encodings(unknown_image)[0]
unknown_encoding_2 = face_encodings(unknown_2)[0]
biden = face_encodings(biden)[0]
biden_unknown = face_encodings(biden_unknown)[0]

results_1 = compare_faces([obama_encoding], unknown_encoding)
results_2 = compare_faces([obama_encoding], unknown_encoding_2)
results_3 = compare_faces([obama_encoding], biden)
results_4 = compare_faces([obama_encoding], biden_unknown)

results_5 = compare_faces([biden], unknown_encoding)
results_6 = compare_faces([biden], unknown_encoding_2)
results_7 = compare_faces([biden], obama_encoding)
results_8 = compare_faces([biden], biden_unknown)
print(results_8)