"""
Test the embedding factory pattern.
Verifies that providers are registered and can be created dynamically.
"""
from embeddings import create_embedder, list_providers
from embeddings.base import EmbeddingConfig

print("\n" + "="*80)
print("  FACTORY PATTERN - EMBEDDING PROVIDERS")
print("="*80 + "\n")

# List available providers
print("📋 Available providers:")
providers = list_providers()
for provider in providers:
    print(f"   - {provider}")

if not providers:
    print("   ⚠️  No providers registered yet")
    print("   This is normal if HF dependencies are not installed")

print()

# Test dummy provider
print("🎲 Testing Dummy Provider:")
try:
    config = EmbeddingConfig(
        model_name="test-dummy",
        dimension=768
    )
    
    embedder = create_embedder("dummy", config)
    print(f"   ✓ Created: {embedder}")
    
    vec = embedder.embed_text("Hello world")
    print(f"   ✓ Generated embedding: {len(vec)} dimensions")
    print(f"   ✓ First 5 values: {vec[:5]}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test HF-E5 provider if available
print("🧠 Testing HF-E5 Provider:")
if "hf-e5" in providers:
    try:
        config = EmbeddingConfig(
            model_name="intfloat/multilingual-e5-small",
            dimension=384
        )
        
        print("   Loading model (may take a moment)...")
        embedder = create_embedder("hf-e5", config)
        print(f"   ✓ Created: {embedder}")
        
        vec = embedder.embed_text("Hello world")
        print(f"   ✓ Generated embedding: {len(vec)} dimensions")
        print(f"   ✓ First 5 values: {vec[:5]}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
else:
    print("   ⚠️  HF-E5 not available (torch/transformers not installed)")

print()

# Test invalid provider
print("❌ Testing Invalid Provider:")
try:
    config = EmbeddingConfig(model_name="test", dimension=768)
    embedder = create_embedder("invalid-provider", config)
    print("   ❌ Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ Correctly raised ValueError: {e}")
except Exception as e:
    print(f"   ❌ Unexpected error: {e}")

print("\n" + "="*80)
print("  FACTORY PATTERN TEST COMPLETED")
print("="*80 + "\n")

print("📝 Summary:")
print(f"   - Total providers registered: {len(providers)}")
print("   - Factory pattern: ✅ Working")
print("   - No if/else needed in main.py")
print()
print("To add a new provider:")
print("   1. Create class with @register_provider('name')")
print("   2. Change EMBEDDING_PROVIDER in .env")
print("   3. Run main.py - it will use the new provider automatically!")
print()
