use crate::tokenizer::{Decoder, Result};

#[derive(Clone, Debug, Default)]
/// ByteFallback is a simple trick which converts tokens looking like `<0x61>`
/// to pure bytes, and attempts to make them into a string. If the tokens
/// cannot be decoded you will get � instead for each inconvertible byte token
#[non_exhaustive]
pub struct ByteFallback {}

impl ByteFallback {
    pub fn new() -> Self {
        Self {}
    }
}

impl Decoder for ByteFallback {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let mut new_tokens: Vec<String> = vec![];
        let mut previous_byte_tokens: Vec<u8> = vec![];

        for token in tokens {
            let bytes = if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                u8::from_str_radix(&token[3..5], 16).ok()
            } else {
                None
            };
            if let Some(bytes) = bytes {
                previous_byte_tokens.push(bytes);
            } else {
                if !previous_byte_tokens.is_empty() {
                    new_tokens.extend(decode_bytes(&previous_byte_tokens));
                    previous_byte_tokens.clear();
                }
                new_tokens.push(token);
            }
        }
        if !previous_byte_tokens.is_empty() {
            new_tokens.extend(decode_bytes(&previous_byte_tokens));
        }

        Ok(new_tokens)
    }
}

/// Decode one run of `<0xNN>` bytes, preserving every maximal valid UTF-8
/// prefix and emitting one `U+FFFD` per byte that is genuinely invalid or
/// left over when the run ends mid-character.
///
/// The previous whole-run `String::from_utf8` attempt discarded a valid
/// character whenever a later byte in the same run was invalid: e.g. the
/// bytes `F0 A0 80 80 E4` (U+20000 plus a stray lead byte) decoded to five
/// replacement characters instead of the character plus one replacement.
/// Walking the run with `Utf8Error::valid_up_to` / `error_len` keeps the
/// valid prefixes and only replaces the bytes UTF-8 itself rejects, which
/// also restores prefix-monotonicity for streaming decodes: a character
/// already emitted from a valid prefix is still the prefix of the next
/// decode, instead of turning into replacement characters.
fn decode_bytes(bytes: &[u8]) -> Vec<String> {
    let mut decoded = Vec::new();
    let mut rest = bytes;
    while !rest.is_empty() {
        match std::str::from_utf8(rest) {
            Ok(valid) => {
                decoded.push(valid.to_owned());
                break;
            }
            Err(error) => {
                let valid_up_to = error.valid_up_to();
                if valid_up_to > 0 {
                    decoded.push(
                        std::str::from_utf8(&rest[..valid_up_to])
                            .expect("valid_up_to is a UTF-8 boundary")
                            .to_owned(),
                    );
                }
                match error.error_len() {
                    Some(len) => {
                        for _ in 0..len {
                            decoded.push("\u{fffd}".into());
                        }
                        rest = &rest[valid_up_to + len..];
                    }
                    None => {
                        // The run ends mid-character: one replacement per
                        // remaining byte, as the previous per-byte fallback
                        // produced for a truncated character.
                        for _ in &rest[valid_up_to..] {
                            decoded.push("\u{fffd}".into());
                        }
                        break;
                    }
                }
            }
        }
    }
    decoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode() {
        let decoder = ByteFallback::new();
        let res = decoder
            .decode_chain(vec!["Hey".into(), "friend!".into()])
            .unwrap();
        assert_eq!(res, vec!["Hey", "friend!"]);

        let res = decoder.decode_chain(vec!["<0x61>".into()]).unwrap();
        assert_eq!(res, vec!["a"]);

        let res = decoder.decode_chain(vec!["<0xE5>".into()]).unwrap();
        assert_eq!(res, vec!["�"]);

        let res = decoder
            .decode_chain(vec!["<0xE5>".into(), "<0x8f>".into()])
            .unwrap();
        assert_eq!(res, vec!["�", "�"]);

        // 叫
        let res = decoder
            .decode_chain(vec!["<0xE5>".into(), "<0x8f>".into(), "<0xab>".into()])
            .unwrap();
        assert_eq!(res, vec!["叫"]);

        let res = decoder
            .decode_chain(vec![
                "<0xE5>".into(),
                "<0x8f>".into(),
                "<0xab>".into(),
                "a".into(),
            ])
            .unwrap();
        assert_eq!(res, vec!["叫", "a"]);

        let res = decoder
            .decode_chain(vec!["<0xE5>".into(), "<0x8f>".into(), "a".into()])
            .unwrap();
        assert_eq!(res, vec!["�", "�", "a"]);
    }

    #[test]
    fn decode_keeps_valid_characters_before_an_invalid_byte() {
        let decoder = ByteFallback::new();

        // U+20000 (`F0 A0 80 80`) followed by a stray lead byte: the valid
        // character survives and only the stray byte is replaced.
        let res = decoder
            .decode_chain(vec![
                "<0xF0>".into(),
                "<0xA0>".into(),
                "<0x80>".into(),
                "<0x80>".into(),
                "<0xE4>".into(),
                "ok".into(),
            ])
            .unwrap();
        assert_eq!(res, vec!["\u{20000}", "\u{fffd}", "ok"]);

        // An invalid byte in the middle of a run: valid bytes on both sides
        // are preserved (`C3 A9` is U+00E9, `61` is `a`).
        let res = decoder
            .decode_chain(vec![
                "<0xC3>".into(),
                "<0xA9>".into(),
                "<0xFF>".into(),
                "<0x61>".into(),
            ])
            .unwrap();
        assert_eq!(res, vec!["é", "\u{fffd}", "a"]);

        // The same, with the invalid byte first.
        let res = decoder
            .decode_chain(vec![
                "<0xFF>".into(),
                "<0xF0>".into(),
                "<0xA0>".into(),
                "<0x80>".into(),
                "<0x80>".into(),
            ])
            .unwrap();
        assert_eq!(res, vec!["\u{fffd}", "\u{20000}"]);

        // A stray continuation byte is replaced; the character after it is kept.
        let res = decoder
            .decode_chain(vec!["<0x80>".into(), "<0xC3>".into(), "<0xA9>".into()])
            .unwrap();
        assert_eq!(res, vec!["\u{fffd}", "é"]);

        // A run that ends mid-character still yields one replacement per
        // remaining byte, as before: a truncated U+20000 is three bytes.
        let res = decoder
            .decode_chain(vec!["<0xF0>".into(), "<0xA0>".into(), "<0x80>".into()])
            .unwrap();
        assert_eq!(res, vec!["\u{fffd}", "\u{fffd}", "\u{fffd}"]);
    }
}
