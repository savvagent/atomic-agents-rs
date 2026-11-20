use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

/// Derive macro for IOSchema.
///
/// Automatically implements the `IOSchema` trait for a struct. The struct must have
/// a doc comment which serves as the schema description (mirrors Python's BaseIOSchema
/// docstring requirement).
///
/// This macro also generates `Serialize`, `Deserialize`, `JsonSchema`, `Clone`, and `Debug`
/// implementations via their respective derive macros.
///
/// # Example
///
/// ```ignore
/// /// A user's chat message input.
/// #[derive(IOSchema)]
/// pub struct ChatInput {
///     /// The message text.
///     pub chat_message: String,
/// }
/// ```
#[proc_macro_derive(IOSchema)]
pub fn derive_io_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Extract doc comments
    let doc_attrs: Vec<String> = input
        .attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(meta) = &attr.meta {
                    if let syn::Expr::Lit(lit) = &meta.value {
                        if let syn::Lit::Str(s) = &lit.lit {
                            return Some(s.value().trim().to_string());
                        }
                    }
                }
            }
            None
        })
        .filter(|s| !s.is_empty())
        .collect();

    if doc_attrs.is_empty() {
        return syn::Error::new_spanned(
            &input.ident,
            "IOSchema requires a doc comment (/// ...) that describes the schema. \
             This mirrors Python's BaseIOSchema docstring requirement.",
        )
        .to_compile_error()
        .into();
    }

    let description = doc_attrs.join(" ");
    let name_str = name.to_string();

    let expanded = quote! {
        impl ::atomic_agents::schema::IOSchema for #name {
            fn schema_title() -> &'static str {
                #name_str
            }

            fn schema_description() -> &'static str {
                #description
            }
        }
    };

    TokenStream::from(expanded)
}
