//! This crate implements array map or vec mac using eytzinger search algorithm.
//! Note that these maps does not support insert/remove operations due to cost.

use eytzinger::SliceExt;
use std::borrow::Borrow;

pub trait AsSlice {
    type Key: Ord;
    type Value;

    fn as_slice<'a>(&'a self) -> &'a [(Self::Key, Self::Value)];
}

pub trait AsMutSlice: AsSlice {
    fn as_mut_slice(&mut self) -> &mut [(Self::Key, Self::Value)];
}

impl<K: Ord, V, const LEN: usize> AsSlice for [(K, V); LEN] {
    type Key = K;
    type Value = V;

    fn as_slice(&self) -> &[(Self::Key, Self::Value)] {
        self
    }
}

impl<K: Ord, V, const LEN: usize> AsMutSlice for [(K, V); LEN] {
    fn as_mut_slice(&mut self) -> &mut [(Self::Key, Self::Value)] {
        self
    }
}

impl<K: Ord, V> AsSlice for &[(K, V)] {
    type Key = K;
    type Value = V;

    fn as_slice(&self) -> &[(Self::Key, Self::Value)] {
        self
    }
}

impl<K: Ord, V> AsSlice for &mut [(K, V)] {
    type Key = K;
    type Value = V;

    fn as_slice(&self) -> &[(Self::Key, Self::Value)] {
        self
    }
}

impl<K: Ord, V> AsMutSlice for &mut [(K, V)] {
    fn as_mut_slice(&mut self) -> &mut [(Self::Key, Self::Value)] {
        self
    }
}

impl<K: Ord, V> AsSlice for Vec<(K, V)> {
    type Key = K;
    type Value = V;

    fn as_slice(&self) -> &[(Self::Key, Self::Value)] {
        self
    }
}

impl<K: Ord, V> AsMutSlice for Vec<(K, V)> {
    fn as_mut_slice(&mut self) -> &mut [(Self::Key, Self::Value)] {
        self
    }
}

/// A map based on a generic slice-compatible type with Eytzinger binary search.
///
/// # Examples
///
/// ```
/// use eytzinger_map::EytzingerMap;
///
/// // `EytzingerMap` doesn't have insert. Build one from another map.
/// let mut movie_reviews = std::collections::BTreeMap::new();
///
/// // review some movies.
/// movie_reviews.insert("Office Space",       "Deals with real issues in the workplace.");
/// movie_reviews.insert("Pulp Fiction",       "Masterpiece.");
/// movie_reviews.insert("The Godfather",      "Very enjoyable.");
///
/// let movie_reviews: EytzingerMap<_> = movie_reviews.into_iter().collect();
///
/// // check for a specific one.
/// if !movie_reviews.contains_key("Les Misérables") {
///     println!("We've got {} reviews, but Les Misérables ain't one.",
///              movie_reviews.len());
/// }
///
/// // look up the values associated with some keys.
/// let to_find = ["Up!", "Office Space"];
/// for movie in &to_find {
///     match movie_reviews.get(movie) {
///        Some(review) => println!("{}: {}", movie, review),
///        None => println!("{} is unreviewed.", movie)
///     }
/// }
///
/// // Look up the value for a key (will panic if the key is not found).
/// println!("Movie review: {}", movie_reviews["Office Space"]);
///
/// // iterate over everything.
/// for (movie, review) in movie_reviews.iter() {
///     println!("{}: \"{}\"", movie, review);
/// }
/// ```
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct EytzingerMap<S>(S);

impl<S, K, V> AsRef<[(K, V)]> for EytzingerMap<S>
where
    S: AsRef<[(K, V)]>,
{
    fn as_ref(&self) -> &[(K, V)] {
        self.0.as_ref()
    }
}

impl<S, K, V> AsMut<[(K, V)]> for EytzingerMap<S>
where
    K: Ord,
    S: AsMut<[(K, V)]>,
{
    fn as_mut(&mut self) -> &mut [(K, V)] {
        self.0.as_mut()
    }
}

impl<Q: ?Sized, S> std::ops::Index<&Q> for EytzingerMap<S>
where
    S::Key: Borrow<Q> + Ord,
    Q: Ord,
    S: AsSlice,
{
    type Output = S::Value;

    /// Returns a reference to the value corresponding to the supplied key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present in the `BTreeMap`.
    #[inline]
    fn index(&self, key: &Q) -> &S::Value {
        self.get(key).expect("no entry found for key")
    }
}

impl<S, K, V> IntoIterator for EytzingerMap<S>
where
    S: std::iter::IntoIterator<Item = (K, V)>,
{
    type Item = (K, V);
    type IntoIter = S::IntoIter;

    fn into_iter(self) -> S::IntoIter {
        self.0.into_iter()
    }
}

impl<K, V> std::iter::FromIterator<(K, V)> for EytzingerMap<Vec<(K, V)>>
where
    K: Ord,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let items: Vec<_> = iter.into_iter().collect();
        Self::new(items)
    }
}

impl<S> EytzingerMap<S>
where
    S: AsSlice,
{
    pub fn new(mut s: S) -> Self
    where
        S: AsMutSlice,
    {
        s.as_mut_slice().sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Self::from_sorted(s)
    }

    pub fn from_sorted(mut s: S) -> Self
    where
        S: AsSlice + AsMutSlice,
    {
        s.as_mut_slice()
            .eytzingerize(&mut eytzinger::permutation::InplacePermutator);
        Self(s)
    }

    pub fn from_eytzingerized(s: S) -> Self
    where
        S: AsSlice,
    {
        Self(s)
    }

    #[inline]
    fn find<Q: ?Sized>(&self, key: &Q) -> Option<usize>
    where
        S::Key: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.0
            .as_slice()
            .eytzinger_search_by(|x| x.0.borrow().cmp(key))
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let map = EytzingerMap::new(vec![(1, "a")]);
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&S::Value>
    where
        S::Key: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.find(key)
            .map(|i| &unsafe { self.0.as_slice().get_unchecked(i) }.1)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let map = EytzingerMap::new(vec![(1, "a")]);
    /// assert_eq!(map.get_key_value(&1), Some(&(1, "a")));
    /// assert_eq!(map.get_key_value(&2), None);
    /// ```
    pub fn get_key_value<Q: ?Sized>(&self, key: &Q) -> Option<&(S::Key, S::Value)>
    where
        S::Key: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.find(key)
            .map(|i| unsafe { self.0.as_slice().get_unchecked(i) })
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let map = EytzingerMap::new(vec![(1, "a")]);
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        S::Key: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.find(key).is_some()
    }

    // range, range_mut

    /// Gets an iterator over the entries of the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let map = EytzingerMap::new(vec![(3, "c"), (2, "b"), (1, "a")]);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter(&self) -> impl std::iter::Iterator<Item = &(S::Key, S::Value)> {
        self.0.as_slice().iter()
    }

    /// Gets an iterator over the keys of the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let map = EytzingerMap::new(vec![(2, "b"), (1, "a")]);
    ///
    /// for key in map.keys() {
    ///     println!("{}", key);
    /// }
    /// ```
    pub fn keys(&self) -> impl std::iter::Iterator<Item = &S::Key> {
        self.iter().map(|(k, _v)| k)
    }

    /// Gets an iterator over the values of the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let map = EytzingerMap::new(vec![(1, "hello"), (2, "goodbye")]);
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    pub fn values(&self) -> impl std::iter::Iterator<Item = &S::Value> {
        self.iter().map(|(_k, v)| v)
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let mut a = EytzingerMap::new(vec![]);
    /// assert_eq!(a.len(), 0);
    /// a = EytzingerMap::new(vec![(1, "a")]);
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.0.as_slice().len()
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let mut a = EytzingerMap::new(vec![]);
    /// assert!(a.is_empty());
    /// a = EytzingerMap::new(vec![(1, "a")]);
    /// assert!(!a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.0.as_slice().is_empty()
    }
}

impl<S> EytzingerMap<S>
where
    S: AsMutSlice,
{
    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let mut map = EytzingerMap::new(vec![(1, "a")]);
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut S::Value>
    where
        S::Key: Borrow<Q> + Ord,
        Q: Ord,
    {
        let i = self.find(key)?;
        Some(&mut unsafe { self.0.as_mut_slice().get_unchecked_mut(i) }.1)
    }

    /// Gets a mutable iterator over the entries of the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let mut map = EytzingerMap::new(vec![("a", 1), ("b", 2), ("c", 3)]);
    ///
    /// // Update all values
    /// for (_, val) in map.iter_mut() {
    ///     *val *= 2;
    /// }
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> impl std::iter::Iterator<Item = &mut (S::Key, S::Value)> {
        self.0.as_mut_slice().iter_mut()
    }

    /// Gets a mutable iterator over the values of the map.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use eytzinger_map::EytzingerMap;
    ///
    /// let mut map = EytzingerMap::new(vec![("a", 1), ("b", 2), ("c", 3)]);
    ///
    /// for val in map.values_mut() {
    ///     *val = *val + 10;
    /// }
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    pub fn values_mut(&mut self) -> impl std::iter::Iterator<Item = &mut S::Value> {
        self.iter_mut().map(|(_k, v)| v)
    }
}

/// A map based on an array with Eytzinger binary search.
/// See [EytzingerMap](eytzinger-map::EytzingerMap) for details.
///
/// ```
/// use eytzinger_map::EytzingerArrayMap;
///
/// let map = EytzingerArrayMap::new([(1, "a"), (2, "b"), (3, "c")]);
/// assert_eq!(map[&1], "a");
/// ```
pub type EytzingerArrayMap<K, V, const LEN: usize> = EytzingerMap<[(K, V); LEN]>;
/// A map based on a Vec with Eytzinger binary search.
/// See [EytzingerMap](eytzinger-map::EytzingerMap) for details.
///
/// ```
/// use eytzinger_map::EytzingerVecMap;
///
/// let map = EytzingerVecMap::new(vec![(1, "a"), (2, "b"), (3, "c")]);
/// assert_eq!(map[&1], "a");
/// ```
pub type EytzingerVecMap<K, V> = EytzingerMap<Vec<(K, V)>>;
/// A map based on a slice ref with Eytzinger binary search.
/// See [EytzingerMap](eytzinger-map::EytzingerMap) for details.
///
/// This is useful when the base data is owned by other data.
/// ```
/// use eytzinger_map::EytzingerRefMap;
///
/// let mut vec = vec![(1, "a"), (2, "b"), (3, "c")];
/// let map = EytzingerRefMap::from_ref(vec.as_mut_slice());
/// assert_eq!(map[&1], "a");
/// ```
pub type EytzingerRefMap<'a, K, V> = EytzingerMap<&'a [(K, V)]>;

impl<'a, K, V> EytzingerRefMap<'a, K, V>
where
    K: Ord,
{
    pub fn from_sorted_ref(s: &'a mut [(K, V)]) -> Self {
        s.eytzingerize(&mut eytzinger::permutation::InplacePermutator);
        Self(s)
    }

    pub fn from_ref(s: &'a mut [(K, V)]) -> Self {
        s.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Self::from_sorted_ref(s)
    }
}
