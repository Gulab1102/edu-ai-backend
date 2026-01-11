from engine import TeachingEngine

engine = TeachingEngine("vector_db/class10_maths_quadratic")

while True:
    q = input("\nAsk (exit): ")
    if q == "exit":
        break
    print(engine.handle(q))
